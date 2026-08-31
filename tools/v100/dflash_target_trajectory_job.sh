#!/usr/bin/env bash
# Capture TurboMind target-model boundaries through layers 0..5 on TP4 V100.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-target-trajectory-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}/parity"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)

TM_DFLASH_TARGET_TRAJECTORY=1 \
    TM_DFLASH_TARGET_PARITY_DIR="${RESULTS}/parity" \
    python3 /job/bench_decode.py "${common[@]}" --output-tokens 64 --trials 1 \
    --json-out "${RESULTS}/trace.json" 2>&1 | tee "${RESULTS}/trace.log"

python3 - "${RESULTS}/parity/target-prompt/lmdeploy" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
dirs = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
assert len(dirs) == 4, dirs
for directory in dirs:
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    matching = [record for record in records if record["name"] == "target.trajectory"]
    assert len(matching) == 1, (directory, len(matching))
    record = matching[0]
    assert record["dtype"] == "f16", record
    assert record["shape"] == [38, 5120], record
    assert record["position"] == 999, record
    assert record["token_id"] == 198, record
    assert record["input_length"] == 1000, record
    assert (directory / record["file"]).stat().st_size == 38 * 5120 * 2
    ids = [item for item in records if item["name"] == "target.input_ids"]
    assert len(ids) == 1 and ids[0]["shape"] == [1000], (directory, ids)
    assert (directory / ids[0]["file"]).stat().st_size == 1000 * 4
    features = [item for item in records if item["name"] == "target.prompt_features"]
    assert len(features) == 1 and features[0]["shape"] == [5, 5120], (directory, features)
    assert (directory / features[0]["file"]).stat().st_size == 5 * 5120 * 2
print(f"DFLASH_TARGET_TRAJECTORY_TRACE_PASS ranks={len(dirs)}")
PY

SG_REF=""
while IFS= read -r candidate; do
    if grep -q 'target.prompt_features' "${candidate}"/rank-*/manifest.jsonl 2>/dev/null; then
        SG_REF="${candidate}"
        break
    fi
done < <(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' -print | sort -r)
[ -n "${SG_REF}" ] || {
    echo 'FAIL: no SGLang target prompt-feature trace exists' >&2
    exit 4
}
python3 /job/compare_dflash_target_features.py \
    --lmdeploy "${RESULTS}/parity/target-prompt/lmdeploy" --sglang "${SG_REF}" \
    --output "${RESULTS}/target_features.json" | tee "${RESULTS}/target_features.log"

TM_DFLASH_TARGET_TRAJECTORY=0 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo DFLASH_TARGET_TRAJECTORY_COMPLETE
