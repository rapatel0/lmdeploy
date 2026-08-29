#!/usr/bin/env bash
# Compare ordinary and per-layer graph target attention on one TP4 DFlash wheel.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-target-attention-graph-${SRC_COMMIT}
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
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

run_arm() {
    local name=$1
    local graph=$2
    local legacy_frontier=$3
    echo "=== ${name}: TARGET_ATTENTION_GRAPH=${graph} LEGACY_FRONTIER_READBACK=${legacy_frontier} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_PAGED_Q8=1 \
        TM_DFLASH_DRAFT_GRAPH=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_TARGET_ATTENTION_GRAPH="${graph}" \
        TM_DFLASH_LEGACY_FRONTIER_READBACK="${legacy_frontier}" \
        TM_DFLASH_GRAPH_TRACE="${graph}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 5 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "${RESULTS}/${name}.json"
}
run_arm legacy_frontier 0 1
run_arm host_frontier 0 0
run_arm target_graph 1 0

if grep -q 'missing host frontier' "${RESULTS}/console.log"; then
    echo "FAIL: host-frontier path used the safety fallback" >&2
    exit 3
fi

capture_count="$(grep -c '\[DFlash2\] target attention graph captured phase=' "${RESULTS}/console.log" || true)"
replay_count="$(grep -c '\[DFlash2\] target attention graph replay phase=' "${RESULTS}/console.log" || true)"
# This audited Qwen3.8 target has four full-attention layers on each of four TP ranks.
expected_graphs="${TM_DFLASH_EXPECTED_TARGET_GRAPHS:-16}"
echo "TARGET_ATTENTION_GRAPH_COUNTS captured=${capture_count} replay=${replay_count} expected=${expected_graphs}"
[ "${capture_count}" -ge "${expected_graphs}" ] || {
    echo "FAIL: expected every TP-rank/layer target attention graph to capture; " \
        "wanted ${expected_graphs}, got ${capture_count}" >&2
    exit 3
}
[ "${replay_count}" -ge "${expected_graphs}" ] || {
    echo "FAIL: expected every TP-rank/layer target attention graph to replay; " \
        "wanted ${expected_graphs}, got ${replay_count}" >&2
    exit 3
}

NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then NSYS=/opt/nsys/nsys; fi
[ -n "${NSYS}" ] || {
    echo "FAIL: nsys unavailable" >&2
    exit 2
}
export FT_NVTX=ON
for arm in "host_frontier 0" "target_graph 1"; do
    read -r name graph <<<"${arm}"
    echo "=== profile ${name} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_PAGED_Q8=1 \
        TM_DFLASH_DRAFT_GRAPH=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_TARGET_ATTENTION_GRAPH="${graph}" \
        TM_DFLASH_LEGACY_FRONTIER_READBACK=0 \
        TM_DFLASH_GRAPH_TRACE="${graph}" \
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
        "${RESULTS}/profile_${name}.nsys-rep" >"${RESULTS}/profile_${name}_stats.log" 2>&1 || {
        echo "WARN: nsys stats failed for ${name}"
        tail -40 "${RESULTS}/profile_${name}_stats.log"
    }
done
unset FT_NVTX

echo "=== target attention graph exact identity ==="
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=1 \
    TM_DFLASH_PAGED_Q8=1 \
    TM_DFLASH_DRAFT_GRAPH=1 \
    TM_DFLASH_SELECTOR_GRAPH=0 \
    TM_DFLASH_TARGET_ATTENTION_GRAPH=1 \
    TM_DFLASH_LEGACY_FRONTIER_READBACK=0 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

touch "${RESULTS}/completed"
echo DFLASH_TARGET_ATTENTION_GRAPH_COMPLETE
