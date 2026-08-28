#!/usr/bin/env bash
# Prove that TurboMind consumes the separate DFlash2 checkpoint and keeps the target baseline coherent.
set -uo pipefail
MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
DRAFT_MODEL="${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-load-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT
cat /src/SOURCE_STAMP
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd / || exit $?
export TM_LOG_LEVEL=INFO
python3 /job/verify_dflash_load.py \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --tp "${TP:-4}"
