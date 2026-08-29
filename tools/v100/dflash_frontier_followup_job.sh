#!/usr/bin/env bash
set -euo pipefail
COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${COMMIT}" ] || COMMIT=unknown
RESULTS="/results/$(date +%Y%m%d_%H%M%S)-dflash-frontier-followup-${COMMIT}"
mkdir -p "${RESULTS}"
exec > >(tee "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; if [ "$rc" -ne 0 ]; then touch "${RESULTS}/incomplete"; fi; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo "FAIL: no prebuilt wheel" >&2; exit 2; }
echo "commit=${COMMIT} wheel=${WHEEL}"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO FT_NVTX=ON
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then NSYS=/opt/nsys/nsys; fi
[ -n "${NSYS}" ] || { echo "FAIL: nsys unavailable" >&2; exit 2; }

profile_arm() {
    local name=$1
    local legacy_frontier=$2
    echo "=== profile ${name} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_PAGED_Q8=1 \
        TM_DFLASH_DRAFT_GRAPH=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_LEGACY_FRONTIER_READBACK="${legacy_frontier}" \
        "${NSYS}" profile \
        --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="${RESULTS}/profile_${name}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 128 --trials 1 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --cuda-profiler-range \
        --json-out "${RESULTS}/profile_${name}.json"
    "${NSYS}" stats \
        --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv --output "${RESULTS}/profile_${name}_stats" \
        "${RESULTS}/profile_${name}.nsys-rep" >"${RESULTS}/profile_${name}_stats.log" 2>&1
}
profile_arm legacy_frontier 1
profile_arm host_frontier 0
unset FT_NVTX

if grep -q 'missing host frontier' "${RESULTS}/console.log"; then
    echo "FAIL: host-frontier path used the safety fallback" >&2
    exit 3
fi

echo "=== host-frontier exact identity ==="
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=1 \
    TM_DFLASH_PAGED_Q8=1 \
    TM_DFLASH_DRAFT_GRAPH=1 \
    TM_DFLASH_SELECTOR_GRAPH=0 \
    TM_DFLASH_LEGACY_FRONTIER_READBACK=0 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

touch "${RESULTS}/completed"
echo DFLASH_FRONTIER_FOLLOWUP_COMPLETE
