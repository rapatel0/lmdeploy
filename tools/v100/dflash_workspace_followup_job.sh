#!/usr/bin/env bash
# Reuse the fd13f231 wheel to classify the known audited near-tie and validate parity capture.
set -euo pipefail

RUNTIME_COMMIT=fd13f2316c8f
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-workspace-followup-${RUNTIME_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

echo "runtime wheel commit=${RUNTIME_COMMIT}"
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

identity_fail=0
for workspace in 0 1; do
    for trial in 1 2; do
        log="${RESULTS}/identity_workspace${workspace}_trial${trial}.log"
        echo "=== identity workspace=${workspace} trial=${trial} ==="
        set +e
        TM_DFLASH_PERSISTENT_WORKSPACE="${workspace}" \
            python3 /job/verify_dflash_audited.py \
            --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
            --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
            --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
            --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
            2>&1 | tee "${log}"
        rc=${PIPESTATUS[0]}
        set -e
        if [ "${rc}" -ne 0 ]; then
            if grep -q '^DFLASH_AUDITED_IDENTITY_FAIL first_difference=220$' "${log}"; then
                echo "IDENTITY_KNOWN_NEAR_TIE workspace=${workspace} trial=${trial}"
            else
                echo "IDENTITY_UNEXPECTED_FAIL workspace=${workspace} trial=${trial} rc=${rc}"
                identity_fail=1
            fi
        fi
    done
done
[ "${identity_fail}" -eq 0 ]

echo "=== persistent workspace parity trace smoke ==="
mkdir -p "${RESULTS}/parity"
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_PARITY_DIR="${RESULTS}/parity" \
    python3 /job/bench_decode.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
    --num-draft-tokens 7 --speculative-algorithm dflash2 \
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 64 --trials 1 \
    --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 \
    --json-out "${RESULTS}/parity_smoke.json"

python3 - "${RESULTS}/parity" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "lmdeploy"
dirs = sorted(path for path in root.glob("rank-*-device-*-pid-*") if path.is_dir())
assert len(dirs) == 4, f"expected four TP trace directories, got {dirs}"
required = {
    "target.post_layer_residual",
    "context.fc",
    "context.norm",
    "block.anchors",
    "block.ids",
    "selector.candidate_ids",
    "selector.unary_scores",
    "selector.scores",
    "selector.selected_ids",
}
for directory in dirs:
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    assert records, f"empty manifest: {directory}"
    assert len({record["ordinal"] for record in records}) == len(records)
    names = {record["name"] for record in records}
    assert required <= names, f"missing {sorted(required - names)} from {directory}"
    for record in records:
        data = directory / record["file"]
        assert data.stat().st_size == record["bytes"], (data, data.stat().st_size, record["bytes"])
print(f"DFLASH_PARITY_TRACE_PASS ranks={len(dirs)}")
PY

touch "${RESULTS}/completed"
echo DFLASH_WORKSPACE_FOLLOWUP_COMPLETE
