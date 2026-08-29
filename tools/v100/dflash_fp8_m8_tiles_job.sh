#!/usr/bin/env bash
# Test opt-in SM70 FP8 M=8 GEMM tiles with grouped DFlash target attention.
# Launch with:
#   tools/v100/run_island2.sh tools/v100/dflash_fp8_m8_tiles_job.sh \
#     lmdeploy-dflash-fp8-m8 tools/v100/bench_decode.py \
#     tools/v100/verify_dflash_audited.py
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-tiles-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

for driver in /job/bench_decode.py /job/verify_dflash_audited.py; do
    [ -f "${driver}" ] || {
        echo "FAIL: missing required driver ${driver}; pass it to run_island2.sh" >&2
        exit 2
    }
done

cat /src/SOURCE_STAMP
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
unset TM_DFLASH_GROUPED_PAGED_Q8

validate_bench() {
    local path=$1
    local trials=$2
    local tokens=$3
    python3 - "${path}" "${trials}" "${tokens}" <<'PY'
import json, sys
path, expected_trials, expected_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = json.load(open(path, encoding="utf-8"))
rows = data.get("trials", [])
if len(rows) != expected_trials:
    raise SystemExit(f"{path}: expected {expected_trials} trials, found {len(rows)}")
for row in rows:
    if row.get("degenerate") or row.get("output_tokens") != expected_tokens:
        raise SystemExit(f"{path}: invalid trial {row}")
PY
}

run_arm() {
    local name=$1
    local candidates=$2
    echo "=== arm=${name} candidates=${candidates} ==="
    local -a env_args=(
        "TM_SM70_FP8_M8_TILE_CANDIDATES=${candidates}"
        "TM_GEMM_TUNE_VERBOSE=1"
    )
    if [ "${candidates}" -eq 1 ]; then
        env_args+=("TM_GEMM_DEBUG_OCCUPANCY=1")
    fi
    env "${env_args[@]}" python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 5 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "${RESULTS}/${name}.json" \
        2>&1 | tee "${RESULTS}/${name}.log"
    validate_bench "${RESULTS}/${name}.json" 5 256
}

run_arm baseline 0
run_arm candidates 1

expected_tp="${TP:-4}"
registration_count="$(grep -c 'SM70_FP8_M8_TILE_CANDIDATES_REGISTERED ' "${RESULTS}/candidates.log" || true)"
[ "${registration_count}" -eq "${expected_tp}" ] || {
    echo "FAIL: expected candidate registration on ${expected_tp} device contexts, got ${registration_count}" >&2
    exit 3
}
for ((device = 0; device < expected_tp; ++device)); do
    grep -q "SM70_FP8_M8_TILE_CANDIDATES_REGISTERED device=${device} " "${RESULTS}/candidates.log" || {
        echo "FAIL: missing candidate registration for CUDA device ${device}" >&2
        exit 3
    }
done
grep -q '\[tune\] device=[0-9].*_8x[0-9][0-9]*x[0-9][0-9]*_.*e4m3' "${RESULTS}/candidates.log" || {
    echo "FAIL: no device-tagged M=8 E4M3 tuner evidence" >&2
    exit 3
}

echo "=== matched Nsight profiles ==="
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then
    NSYS=/opt/nsys/nsys
fi
[ -n "${NSYS}" ] || {
    echo "FAIL: nsys unavailable" >&2
    exit 2
}
export FT_NVTX=ON
for name in baseline candidates; do
    candidates=0
    [ "${name}" = candidates ] && candidates=1
    echo "=== profile arm=${name} candidates=${candidates} ==="
    TM_SM70_FP8_M8_TILE_CANDIDATES="${candidates}" \
        TM_GEMM_TUNE_VERBOSE=1 \
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
        --json-out "${RESULTS}/profile_${name}.json" \
        2>&1 | tee "${RESULTS}/profile_${name}.log"
    validate_bench "${RESULTS}/profile_${name}.json" 1 128
    "${NSYS}" stats \
        --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv --output "${RESULTS}/profile_${name}_stats" \
        "${RESULTS}/profile_${name}.nsys-rep" >"${RESULTS}/profile_${name}_stats.log" 2>&1 || {
        echo "WARN: nsys stats failed for ${name}"
        tail -40 "${RESULTS}/profile_${name}_stats.log"
    }
done
unset FT_NVTX
for name in baseline candidates; do
    grep -q 'MMA_884_GROUPED' "${RESULTS}/profile_${name}_stats_cuda_gpu_kern_sum.csv" || {
        echo "FAIL: grouped Q8H4 attention kernel missing from ${name} profile" >&2
        exit 3
    }
done

python3 /src/tools/v100/analyze_dflash_fp8_m8_tiles.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
grep -q '^DFLASH_FP8_M8_TILE_ANALYSIS ' "${RESULTS}/analysis.log"

echo "=== candidate-tile exact audited identity ==="
TM_SM70_FP8_M8_TILE_CANDIDATES=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

touch "${RESULTS}/completed"
echo DFLASH_FP8_M8_TILE_CANDIDATES_COMPLETE
