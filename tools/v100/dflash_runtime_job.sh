#!/usr/bin/env bash
# Build and run the first end-to-end DFlash2 verification path.
set -uo pipefail
MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
DRAFT_MODEL="${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-runtime-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT
cat /src/SOURCE_STAMP
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -40
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd / || exit $?
export TM_LOG_LEVEL=INFO
python3 /job/verify_dflash_runtime.py \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --tp "${TP:-4}" 2>&1 | tee "${RESULTS}/driver.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
grep -q 'DFLASH_RUNTIME_IDENTITY_PASS' "${RESULTS}/driver.log" || exit 4
echo DFLASH_RUNTIME_COMPLETE
