#!/usr/bin/env bash
# Build and smoke the qualified default V32 GDN path plus the V128 control.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-gdn-default-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
for driver in /job/bench_decode.py /job/verify_dflash_audited.py; do
    [ -f "${driver}" ] || {
        echo "FAIL: missing ${driver}" >&2
        exit 2
    }
done
rm -f /wheels/lmdeploy-*.whl
build_started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] && [ "$(stat -c %Y "${WHEEL}")" -ge "${build_started}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --output-tokens 128 --trials 1 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
python3 /job/bench_decode.py "${common[@]}" --json-out "${RESULTS}/default.json" 2>&1 | tee "${RESULTS}/default.log"
TM_GDN_SM70_VALUE_COLS=128 python3 /job/bench_decode.py "${common[@]}" --json-out "${RESULTS}/legacy.json" \
    2>&1 | tee "${RESULTS}/legacy.log"
for ((device = 0; device < "${TP:-4}"; ++device)); do
    grep -q "GDN_SM70_VALUE_COLS_ACTIVE device=${device} value_cols=32" "${RESULTS}/default.log"
done
if grep -q 'GDN_SM70_VALUE_COLS_ACTIVE' "${RESULTS}/legacy.log"; then
    echo 'FAIL: legacy arm activated a split kernel' >&2
    exit 3
fi
python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_default.log"
touch "${RESULTS}/completed"
echo DFLASH_GDN_DEFAULT_SMOKE_COMPLETE
