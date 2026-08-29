#!/usr/bin/env bash
# Qualify combined Q=8 and GQA-head grouped paged attention on TP4 V100.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-grouped-paged-q8-${SRC_COMMIT}
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
python3 /src/tools/v100/check_dflash_grouped_geometry.py | tee "${RESULTS}/geometry.log"
grep -q '^GROUPED_Q8H4_GEOMETRY_PASS ' "${RESULTS}/geometry.log"
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

run_arm() {
    local name=$1
    local paged=$2
    local grouped=$3
    echo "=== arm=${name} TM_DFLASH_PAGED_Q8=${paged} TM_DFLASH_GROUPED_PAGED_Q8=${grouped} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_DRAFT_GRAPH=0 \
        TM_DFLASH_PAGED_Q8="${paged}" \
        TM_DFLASH_GROUPED_PAGED_Q8="${grouped}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 5 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 \
        --json-out "${RESULTS}/${name}.json"
    python3 - "${RESULTS}/${name}.json" 5 256 <<'PY'
import json, sys
path, expected_trials, expected_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = json.load(open(path, encoding="utf-8"))
trials = data.get("trials", [])
if len(trials) != expected_trials:
    raise SystemExit(f"{path}: expected {expected_trials} trials, found {len(trials)}")
for trial in trials:
    if trial.get("degenerate") or trial.get("output_tokens") != expected_tokens:
        raise SystemExit(f"{path}: invalid trial {trial}")
PY
}

# One build, three attribution arms: qualified flattened default, the existing
# direct-paged prerequisite, and the new grouped direct-paged kernel.
run_arm flattened 0 0
run_arm paged 1 0
run_arm grouped 1 1

run_context_arm() {
    local name=$1
    local input_tokens=$2
    local paged=$3
    local grouped=$4
    local draft_graph=$5
    echo "=== context arm=${name} input=${input_tokens} paged=${paged} grouped=${grouped} graph=${draft_graph} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_DRAFT_GRAPH="${draft_graph}" \
        TM_DFLASH_GRAPH_TRACE="${draft_graph}" \
        TM_DFLASH_PAGED_Q8="${paged}" \
        TM_DFLASH_GROUPED_PAGED_Q8="${grouped}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens "${input_tokens}" --output-tokens 64 --trials 1 \
        --sglang-corpus /sglang-corpus --cache-max-entry-count 0.05 \
        --json-out "${RESULTS}/${name}.json"
    python3 - "${RESULTS}/${name}.json" 1 64 <<'PY'
import json, sys
path, expected_trials, expected_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = json.load(open(path, encoding="utf-8"))
trials = data.get("trials", [])
if len(trials) != expected_trials:
    raise SystemExit(f"{path}: expected {expected_trials} trial, found {len(trials)}")
for trial in trials:
    if trial.get("degenerate") or trial.get("output_tokens") != expected_tokens:
        raise SystemExit(f"{path}: invalid trial {trial}")
PY
}

# Context and graph matrix from the qualification contract. The short arm
# exercises fixed graph overprovisioning when only split zero is live.
run_context_arm short_graph 64 1 1 1
short_graph_captures="$(grep -c '\[DFlash2\] draft graph captured phase=' "${RESULTS}/console.log" || true)"
[ "${short_graph_captures}" -ge 4 ] || {
    echo "FAIL: expected short-context draft graph capture on four TP ranks, got ${short_graph_captures}" >&2
    exit 3
}
run_context_arm flattened_8k 8000 0 0 0
run_context_arm grouped_8k 8000 1 1 0
run_context_arm flattened_25k 25000 0 0 0
run_context_arm grouped_25k 25000 1 1 0

echo "=== 64-instance grouped activation and same-input parity ==="
PARITY_DIR="${RESULTS}/attention-parity"
mkdir -p "${PARITY_DIR}"
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=1 \
    TM_DFLASH_SELECTOR_GRAPH=0 \
    TM_DFLASH_DRAFT_GRAPH=0 \
    TM_DFLASH_PAGED_Q8=1 \
    TM_DFLASH_GROUPED_PAGED_Q8=1 \
    TM_DFLASH_GROUPED_PAGED_Q8_PARITY_DIR="${PARITY_DIR}" \
    python3 /job/bench_decode.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
    --num-draft-tokens 7 --speculative-algorithm dflash2 \
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 32 --trials 1 \
    --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 \
    --json-out "${RESULTS}/parity.json"
python3 /src/tools/v100/verify_dflash_grouped_parity.py "${PARITY_DIR}" | tee "${RESULTS}/parity-validation.log"
grep -q '^GROUPED_Q8H4_PARITY_PASS ' "${RESULTS}/parity-validation.log"

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
for spec in "flattened:0:0" "paged:1:0" "grouped:1:1"; do
    IFS=: read -r name paged grouped <<<"${spec}"
    echo "=== profile arm=${name} paged=${paged} grouped=${grouped} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE=1 \
        TM_DFLASH_ONE_PASS_REJECT=1 \
        TM_DFLASH_SELECTOR_GRAPH=0 \
        TM_DFLASH_DRAFT_GRAPH=0 \
        TM_DFLASH_PAGED_Q8="${paged}" \
        TM_DFLASH_GROUPED_PAGED_Q8="${grouped}" \
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
        echo "WARN: nsys stats failed for arm=${name}"
        tail -40 "${RESULTS}/profile_${name}_stats.log"
    }
done
unset FT_NVTX

echo "=== grouped-Q8H4 exact audited identity ==="
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=1 \
    TM_DFLASH_SELECTOR_GRAPH=0 \
    TM_DFLASH_DRAFT_GRAPH=0 \
    TM_DFLASH_PAGED_Q8=1 \
    TM_DFLASH_GROUPED_PAGED_Q8=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

echo "=== grouped-Q8H4 plus draft-graph exact audited identity ==="
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=1 \
    TM_DFLASH_SELECTOR_GRAPH=0 \
    TM_DFLASH_DRAFT_GRAPH=1 \
    TM_DFLASH_PAGED_Q8=1 \
    TM_DFLASH_GROUPED_PAGED_Q8=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity_graph.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity_graph.log"

touch "${RESULTS}/completed"
echo DFLASH_GROUPED_PAGED_Q8_COMPLETE
