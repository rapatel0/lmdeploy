#!/usr/bin/env bash
# Profile matched target-only and DFlash2 decode requests with Nsight Systems.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-nsys-dflash-${SRC_COMMIT}
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
[ -n "${NSYS}" ] || {
    echo "FAIL: nsys unavailable" >&2
    exit 2
}
"${NSYS}" --version

if [ "${REUSE_WHEEL:-0}" != 1 ]; then
    if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
        grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -40
        exit 2
    fi
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd / || exit $?
export TM_LOG_LEVEL=INFO
export FT_NVTX=ON

for k in ${PROFILE_K_VALUES:-0 7}; do
    echo "=== profile DFlash K=${k} ==="
    args=(
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
        --num-draft-tokens "${k}"
        --input-tokens 1000 --output-tokens 128 --trials 1
        --sglang-corpus /sglang-corpus
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
        --cache-max-entry-count 0.05 --cuda-profiler-range
        --json-out "${RESULTS}/bench_k${k}.json"
    )
    if [ "${k}" -gt 0 ]; then
        args+=(
            --speculative-algorithm dflash2
            --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
            --speculative-dflash-block-size 8
            --speculative-draft-window 2048
        )
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
        --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv \
        --output "${RESULTS}/k${k}_stats" \
        "${RESULTS}/k${k}.nsys-rep" >"${RESULTS}/k${k}_stats.log" 2>&1 || {
        echo "WARN: nsys stats failed for K=${k}"
        tail -40 "${RESULTS}/k${k}_stats.log"
    }
done

touch "${RESULTS}/completed"
echo NSYS_DFLASH_COMPLETE
