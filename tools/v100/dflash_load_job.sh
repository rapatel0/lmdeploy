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
export TM_DFLASH_CAPTURE=1
export TM_DFLASH_TRACE_SELECTOR=1
python3 /job/verify_dflash_load.py \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --tp "${TP:-4}" 2>&1 | tee "${RESULTS}/driver.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
grep -q '\[DFlash2\] target capture shape=.*layer_ids=\[5,19,33,47,61\].*nonzero_bytes=\[[1-9]' \
    "${RESULTS}/driver.log" || {
    echo "FAIL: target residual capture was absent or empty"
    exit 4
}
grep -q '\[DFlash2\] context projection shape=.*nonzero_bytes=[1-9]' "${RESULTS}/driver.log" || {
    echo "FAIL: DFlash2 context projection was absent or empty"
    exit 5
}
grep -q '\[DFlash2\] materialized context KV: tokens=.*layers=5' "${RESULTS}/driver.log" || {
    echo "FAIL: DFlash2 context KV materialization did not run"
    exit 6
}
grep -q '\[DFlash2\] grouped convolution shape=.*nonzero_bytes=[1-9]' "${RESULTS}/driver.log" || {
    echo "FAIL: DFlash2 grouped convolution was absent or empty"
    exit 7
}
grep -q '\[DFlash2\] five-layer draft shape=.*nonzero_bytes=[1-9]' "${RESULTS}/driver.log" || {
    echo "FAIL: DFlash2 five-layer draft did not complete"
    exit 8
}
grep -Eq '\[DFlash2\] greedy candidates=\[([0-9]+,){6}[0-9]+\]' "${RESULTS}/driver.log" || {
    echo "FAIL: DFlash2 candidate selector did not emit seven tokens"
    exit 9
}
echo DFLASH_CAPTURE_PASS
echo DFLASH_CONTEXT_PROJECTION_PASS
echo DFLASH_CONTEXT_KV_PASS
echo DFLASH_GROUPED_CONV_PASS
echo DFLASH_FIVE_LAYER_PASS
echo DFLASH_SELECTOR_PASS
