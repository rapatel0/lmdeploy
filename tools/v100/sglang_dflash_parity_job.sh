#!/usr/bin/env bash
# Trace the first real SGLang DFlash block without modifying SGLang source.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-sglang-dflash-parity-${SRC_COMMIT}
mkdir -p "${RESULTS}/trace"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
server_pid=
finish() {
    rc=$?
    if [ -n "${server_pid}" ]; then kill -- "-${server_pid}" 2>/dev/null || true; fi
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT
cat /src/SOURCE_STAMP
command -v sglang >/dev/null

mkdir -p /tmp/dflash-parity-site
cp /job/sglang_dflash_parity_sitecustomize.py /tmp/dflash-parity-site/sitecustomize.py

if [ "${SGLANG_PARITY_PRODUCTION_CONFIG:-0}" = 1 ]; then
    if [ "${SGLANG_PARITY_DISABLE_CUDA_GRAPH:-0}" = 1 ]; then
        graph_args=(--disable-cuda-graph)
    else
        graph_args=(--cuda-graph-max-bs 8 --cuda-graph-bs 1 2 4 8)
    fi
    runtime_args=(
        --kv-cache-dtype auto
        --context-length 262144
        --max-total-tokens 600000
        --max-running-requests 8
        --max-mamba-cache-size 40
        --enable-nccl-nvls
    )
elif [ "${SGLANG_PARITY_DISABLE_CUDA_GRAPH:-0}" = 1 ]; then
    graph_args=(--disable-cuda-graph)
    runtime_args=(--kv-cache-dtype fp8_e5m2 --context-length 16384 --max-total-tokens 16384 --max-running-requests 1)
else
    graph_args=(--cuda-graph-max-bs 1 --cuda-graph-bs 1)
    runtime_args=(--kv-cache-dtype fp8_e5m2 --context-length 16384 --max-total-tokens 16384 --max-running-requests 1)
fi

wait_for_server() {
    for _ in $(seq 1 900); do
        # Both /health_generate and /health submit or wait on synthetic model
        # work in this pinned image, which reaches the broken V100 accept
        # kernel before the audited trace is armed. CUDA-graph initialization
        # finishes before Uvicorn announces startup, so use that durable log
        # boundary and make the audited request the first post-startup work.
        if grep -q 'Application startup complete' "${RESULTS}/server.log" 2>/dev/null; then return 0; fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "FAIL: SGLang server exited during startup" >&2
            tail -200 "${RESULTS}/server.log" >&2
            return 1
        fi
        sleep 1
    done
    echo "FAIL: SGLang server did not become ready" >&2
    tail -200 "${RESULTS}/server.log" >&2
    return 1
}

setsid env \
    PYTHONPATH=/tmp/dflash-parity-site:/opt/sglang/python \
    SGLANG_DFLASH_PARITY_DIR="${RESULTS}/trace" \
    SGLANG_DFLASH_PARITY_ARM_FILE="${RESULTS}/trace/armed" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    NCCL_P2P_LEVEL=NVL \
    SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
    SGLANG_MAMBA_CONV_DTYPE=float16 \
    SGLANG_MAMBA_SSM_DTYPE=float16 \
    SGLANG_ENABLE_SPEC_V2=1 \
    sglang serve \
    --trust-remote-code \
    --skip-server-warmup \
    --model-path "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --dtype float16 \
    --attention-backend "${SGLANG_PARITY_ATTENTION_BACKEND:-flash_attn_v100}" \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend triton \
    --tensor-parallel-size "${TP:-4}" \
    --host 0.0.0.0 --port 8082 \
    --mem-fraction-static 0.75 \
    --disable-overlap-schedule \
    "${runtime_args[@]}" \
    "${graph_args[@]}" \
    --chunked-prefill-size 8192 \
    --mamba-full-memory-ratio 0.1 --mamba-scheduler-strategy extra_buffer \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 \
    --speculative-draft-model-quantization unquant \
    --speculative-draft-window-size 2048 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    >"${RESULTS}/server.log" 2>&1 &
server_pid=$!
wait_for_server

# Arm the in-memory hooks only after all synthetic server warm-up blocks have
# finished, immediately before the first audited client request.
touch "${RESULTS}/trace/armed"
set +e
python3 /job/sglang_dflash_parity_client.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --corpus /sglang-corpus \
    --prompt-builder /job/bench_decode.py \
    --expected-hash 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --output "${RESULTS}/response.json"
client_rc=$?
set -e
if [ "${client_rc}" -ne 0 ]; then
    echo "WARN: audited request failed after draft replay; validating the flushed first-block trace"
fi

kill -- "-${server_pid}" 2>/dev/null || true
for _ in $(seq 1 60); do
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 1
done
kill -KILL -- "-${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true
server_pid=

python3 - "${RESULTS}/trace/sglang" <<'PY'
import json
import os
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
dirs = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
assert len(dirs) == 4, f"expected four TP trace directories, got {dirs}"
required = {
    "target.post_layer_residual", "context.fc", "context.norm",
    "block.ids", "block.embedding", "layer0.attention.conv_side0",
    "layer4.output.hidden", "selector.candidate_ids", "selector.unary_scores",
    "selector.score_lattice", "selector.selected_ids",
    "layer0.attention.tilelang.q", "layer0.attention.tilelang.k",
    "layer0.attention.tilelang.v", "layer0.attention.tilelang.output",
    "layer0.attention.tilelang.seq_lens", "layer0.attention.tilelang.query_start_loc",
}
for directory in dirs:
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    names = {record["name"] for record in records}
    assert required <= names, f"missing {sorted(required - names)} from {directory}"
    by_name = {record["name"]: record for record in records}
    block_record = by_name["block.ids"]
    block_payload = (directory / block_record["file"]).read_bytes()
    assert block_record["dtype"] == "i64" and len(block_payload) == 64
    block_ids = list(struct.unpack("<8q", block_payload))
    block_index = int(os.environ.get("SGLANG_DFLASH_PARITY_BLOCK_INDEX", "1"))
    if block_index == 1:
        expected_ids = [1144] + [248070] * 7
        assert block_ids == expected_ids, f"non-audited block IDs in {directory}: {block_ids}"
    else:
        assert block_ids[0] != 248070 and block_ids[1:] == [248070] * 7, \
            f"invalid proposal block IDs in {directory}: {block_ids}"
    for record in records:
        data = directory / record["file"]
        assert data.stat().st_size == record["bytes"]
print(f"SGLANG_DFLASH_PARITY_TRACE_PASS ranks={len(dirs)}")
PY

LM_PARITY_REF=${LM_DFLASH_PARITY_REF:-}
if [ -z "${LM_PARITY_REF}" ]; then
    while IFS= read -r candidate; do
        LM_PARITY_REF=${candidate}
    done < <(find /results -maxdepth 4 -type d -path '*/parity/lmdeploy' | sort)
fi
[ -d "${LM_PARITY_REF}" ] || {
    echo "FAIL: LMDeploy parity reference not found: ${LM_PARITY_REF}" >&2
    exit 4
}
python3 /job/compare_dflash_parity.py \
    --lmdeploy "${LM_PARITY_REF}" \
    --sglang "${RESULTS}/trace/sglang" \
    --output "${RESULTS}/cross_runtime_parity.json"

if [ -n "${LM_DFLASH_TARGET_TRAJECTORY_REF:-}" ]; then
    python3 /job/compare_dflash_target_trajectory.py \
        --lmdeploy "${LM_DFLASH_TARGET_TRAJECTORY_REF}" \
        --sglang "${RESULTS}/trace/sglang" \
        --output "${RESULTS}/target_trajectory.json"
fi

touch "${RESULTS}/completed"
echo SGLANG_DFLASH_PARITY_COMPLETE
