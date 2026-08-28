#!/usr/bin/env bash
# Profile one measured K=0/K=1/K=4 request with Nsight Systems.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-nsys-mtp-${SRC_COMMIT}
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
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then
    NSYS=/opt/nsys/nsys
fi
if [ -z "${NSYS}" ]; then
    echo "FAIL: nsys is not installed in the job image or staged at /opt/nsys" >&2
    exit 2
fi
"${NSYS}" --version

if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    echo "FAIL: build" >&2
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -20
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

for k in 0 1 4; do
    echo "=== profile K=${k} ==="
    args=(
        --model "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens "${k}"
        --input-tokens 1024 --output-tokens 128 --trials 1
        --cuda-profiler-range --json-out "${RESULTS}/bench_k${k}.json"
    )
    if [ "${k}" -gt 0 ]; then
        args+=(--require-mtp)
    fi
    "${NSYS}" profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output="${RESULTS}/k${k}" \
        python3 /job/bench_decode.py "${args[@]}" || exit $?

    "${NSYS}" stats \
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv \
        --output "${RESULTS}/k${k}_stats" \
        "${RESULTS}/k${k}.nsys-rep" >"${RESULTS}/k${k}_stats.log" 2>&1 || {
        echo "WARN: nsys stats failed for K=${k}"
        tail -40 "${RESULTS}/k${k}_stats.log"
    }
done

python3 - "${RESULTS}" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
print("=== Nsight summaries ===")
for k in (0, 1, 4):
    print(f"--- K={k} ---")
    paths = sorted(glob.glob(os.path.join(root, f"k{k}_stats*_cuda_*sum*.csv")))
    if not paths:
        print("no CSV summary; inspect", os.path.join(root, f"k{k}_stats.log"))
        continue
    for path in paths:
        print(os.path.basename(path))
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))
        for row in rows[:12]:
            print(",".join(row))
PY

touch "${RESULTS}/completed"
echo NSYS_MTP_COMPLETE
