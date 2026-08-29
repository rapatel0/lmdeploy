#!/usr/bin/env bash
# Profile TurboMind's baseline SM70 block-FP8 M=8 GEMM in a standalone
# single-GPU driver. Keeping NCCL and engine warmup outside NCU avoids
# profiler-induced TP deadlock.
# Launch with:
#   tools/v100/run_island2.sh tools/v100/dflash_fp8_m8_ncu_job.sh \
#     lmdeploy-dflash-fp8-m8-ncu-v2 tools/v100/dflash_fp8_m8_microbench.cu
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-ncu-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

[ -f /job/dflash_fp8_m8_microbench.cu ] || {
    echo "FAIL: missing /job/dflash_fp8_m8_microbench.cu" >&2
    exit 2
}
cat /src/SOURCE_STAMP
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo "FAIL: no built wheel" >&2
    exit 2
}
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

NCU="$(command -v ncu 2>/dev/null || true)"
NVCC="$(command -v nvcc 2>/dev/null || true)"
[ -n "${NCU}" ] || {
    echo "FAIL: ncu unavailable" >&2
    exit 2
}
[ -n "${NVCC}" ] || {
    echo "FAIL: nvcc unavailable" >&2
    exit 2
}
"${NCU}" --version | tee "${RESULTS}/ncu-version.txt"

LIB="$(
    python3 - <<'PY'
import glob, os
paths = glob.glob('/opt/venv/lib/python*/site-packages/lmdeploy/lib/_turbomind*.so')
if not paths:
    raise SystemExit(1)
print(paths[0])
PY
)"
LIBDIR="$(dirname "${LIB}")"
FMT_HEADER="$(find /src/build -path '*/fmt/format.h' -print -quit)"
[ -n "${FMT_HEADER}" ] || {
    echo "FAIL: fmt build header unavailable" >&2
    exit 2
}
FMT_INCLUDE="${FMT_HEADER%/fmt/format.h}"
PYLIB="$(find /usr /opt -name 'libpython3.12.so*' -type f -print -quit 2>/dev/null)"
[ -n "${PYLIB}" ] || {
    echo "FAIL: shared Python library unavailable" >&2
    exit 2
}
printf 'libturbomind=%s\nfmt_include=%s\nlibpython=%s\n' \
    "${LIB}" "${FMT_INCLUDE}" "${PYLIB}" | tee "${RESULTS}/library.txt"

"${NVCC}" -std=c++17 -O2 -lineinfo \
    -gencode arch=compute_70,code=sm_70 \
    -I/src -I"${FMT_INCLUDE}" /job/dflash_fp8_m8_microbench.cu "${LIB}" \
    -Xlinker -rpath -Xlinker "${LIBDIR}" \
    -lcudart -Xlinker "${PYLIB}" -o /tmp/dflash_fp8_m8_microbench \
    >"${RESULTS}/compile.log" 2>&1 || {
    tail -100 "${RESULTS}/compile.log"
    exit 2
}

export TM_SM70_FP8_M8_TILE_CANDIDATES=0
export TM_SM70_FP8_M8_REUSE_SCALE=0
export TM_GEMM_TUNE_VERBOSE=1
/tmp/dflash_fp8_m8_microbench 2>&1 | tee "${RESULTS}/smoke.log"
grep -q '^DFLASH_FP8_M8_MICROBENCH_COMPLETE$' "${RESULTS}/smoke.log"

"${NCU}" \
    --force-overwrite \
    --target-processes all \
    --profile-from-start off \
    --kernel-name-base demangled \
    --kernel-name 'regex:gemm_kernel.*MMA_Map.*int.8.*Operand_B_Pack.*e4m3' \
    --launch-skip 0 \
    --launch-count 3 \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    --section WarpStateStats \
    --export "${RESULTS}/fp8_m8" \
    /tmp/dflash_fp8_m8_microbench \
    2>&1 | tee "${RESULTS}/ncu.log"

[ -f "${RESULTS}/fp8_m8.ncu-rep" ] || {
    echo "FAIL: NCU report missing" >&2
    exit 3
}
"${NCU}" --import "${RESULTS}/fp8_m8.ncu-rep" --csv --page details \
    >"${RESULTS}/fp8_m8_details.csv" 2>"${RESULTS}/fp8_m8_import.log"
grep -q 'GPU Speed Of Light Throughput' "${RESULTS}/fp8_m8_details.csv" || {
    echo "FAIL: expected SpeedOfLight section missing" >&2
    exit 3
}

touch "${RESULTS}/completed"
echo DFLASH_FP8_M8_NCU_COMPLETE
