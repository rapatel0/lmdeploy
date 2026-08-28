#!/usr/bin/env bash
# Compare two-pass ambiguity detection with a one-pass deterministic top-2 reduction.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-one-pass-reject-${SRC_COMMIT}
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
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

run_arm() {
    local name=$1 workspace=$2 one_pass=$3
    echo "=== ${name}: WORKSPACE=${workspace} ONE_PASS=${one_pass} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE="${workspace}" \
        TM_DFLASH_ONE_PASS_REJECT="${one_pass}" \
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
}

run_arm dynamic 0 0
run_arm workspace 1 0
run_arm one_pass 1 1

echo "=== persistent workspace parity trace smoke ==="
mkdir -p "${RESULTS}/parity"
TM_DFLASH_PERSISTENT_WORKSPACE=1 \
    TM_DFLASH_ONE_PASS_REJECT=0 \
    TM_DFLASH_PARITY_DIR="${RESULTS}/parity" \
    python3 /job/bench_decode.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
    --num-draft-tokens 7 --speculative-algorithm dflash2 \
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 64 --trials 1 \
    --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 \
    --json-out "${RESULTS}/parity_smoke.json"

python3 - "${RESULTS}/parity" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "lmdeploy"
dirs = sorted(path for path in root.glob("rank-*-device-*-pid-*") if path.is_dir())
assert len(dirs) == 4, f"expected four TP trace directories, got {dirs}"
required = {
    "target.post_layer_residual",
    "context.fc",
    "context.norm",
    "block.anchors",
    "block.ids",
    "selector.candidate_ids",
    "selector.unary_scores",
    "selector.scores",
    "selector.selected_ids",
}
for directory in dirs:
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    assert records, f"empty manifest: {directory}"
    assert len({record["ordinal"] for record in records}) == len(records)
    names = {record["name"] for record in records}
    assert required <= names, f"missing {sorted(required - names)} from {directory}"
    for record in records:
        data = directory / record["file"]
        assert data.stat().st_size == record["bytes"], (data, data.stat().st_size, record["bytes"])
print(f"DFLASH_PARITY_TRACE_PASS ranks={len(dirs)}")
PY

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
for arm in "dynamic 0 0" "workspace 1 0" "one_pass 1 1"; do
    read -r name workspace one_pass <<<"${arm}"
    echo "=== profile ${name} ==="
    TM_DFLASH_PERSISTENT_WORKSPACE="${workspace}" \
        TM_DFLASH_ONE_PASS_REJECT="${one_pass}" \
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

echo "=== one-pass exact identity ==="
TM_DFLASH_PERSISTENT_WORKSPACE=1 TM_DFLASH_ONE_PASS_REJECT=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

touch "${RESULTS}/completed"
echo DFLASH_ONE_PASS_REJECT_COMPLETE
