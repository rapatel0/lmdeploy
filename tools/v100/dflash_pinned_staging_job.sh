#!/usr/bin/env bash
# Compare pageable, pinned, and pinned-plus-combined DFlash host readbacks.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-pinned-staging-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -60
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

run_arm() {
    local name=$1 pinned=$2 combined=$3
    echo "=== ${name}: PINNED=${pinned} COMBINED=${combined} ==="
    TM_DFLASH_PINNED_STAGING="${pinned}" \
        TM_DFLASH_COMBINE_ROLLBACK_SYNCS="${combined}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 5 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 \
        --json-out "${RESULTS}/${name}.json"
}

run_arm pageable 0 0
run_arm pinned 1 0
run_arm pinned_combined 1 1

echo "=== pinned combined exact identity ==="
TM_DFLASH_PINNED_STAGING=1 TM_DFLASH_COMBINE_ROLLBACK_SYNCS=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

echo "=== first-real-block parity trace smoke ==="
mkdir -p "${RESULTS}/parity"
TM_DFLASH_PINNED_STAGING=1 TM_DFLASH_COMBINE_ROLLBACK_SYNCS=1 \
    TM_DFLASH_PARITY_DIR="${RESULTS}/parity" \
    python3 /job/bench_decode.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
    --num-draft-tokens 7 --speculative-algorithm dflash2 \
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 8 --trials 1 \
    --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 \
    --json-out "${RESULTS}/parity_smoke.json"

python3 - "${RESULTS}/parity" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / 'lmdeploy'
dirs = sorted(path for path in root.glob('rank-*-device-*-pid-*') if path.is_dir())
assert len(dirs) == 4, f'expected four TP trace directories, got {dirs}'
for directory in dirs:
    records = [json.loads(line) for line in (directory / 'manifest.jsonl').read_text().splitlines()]
    assert records, f'empty manifest: {directory}'
    assert len({record['ordinal'] for record in records}) == len(records)
    names = {record['name'] for record in records}
    for required in ('target.post_layer_residual', 'context.fc', 'context.norm',
                     'block.anchors', 'block.ids', 'selector.candidate_ids',
                     'selector.unary_scores', 'selector.scores', 'selector.selected_ids'):
        assert required in names, f'{required} missing from {directory}'
    for record in records:
        data = directory / record['file']
        assert data.stat().st_size == record['bytes'], (data, data.stat().st_size, record['bytes'])
print(f'DFLASH_PARITY_TRACE_PASS ranks={len(dirs)}')
PY

touch "${RESULTS}/completed"
echo DFLASH_PINNED_STAGING_COMPLETE
