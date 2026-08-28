#!/usr/bin/env bash
# Compare SGLang's native NEXTN path with LMDeploy on the same embedded MTP weights.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-sglang-nextn-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
server_pid=
finish() {
    rc=$?
    if [ -n "${server_pid}" ]; then
        kill -- "-${server_pid}" 2>/dev/null || true
    fi
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

command -v sglang >/dev/null 2>&1 || {
    echo "FAIL: sglang is not installed in the selected image" >&2
    exit 2
}
python3 - <<'PY'
import importlib.metadata
print("sglang", importlib.metadata.version("sglang"))
PY

wait_for_server() {
    local log=$1
    for _ in $(seq 1 900); do
        if curl -fsS http://127.0.0.1:8082/health_generate >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "FAIL: SGLang server exited during startup" >&2
            tail -160 "${log}" >&2
            return 1
        fi
        sleep 1
    done
    echo "FAIL: SGLang server did not become ready" >&2
    tail -160 "${log}" >&2
    return 1
}

stop_server() {
    if [ -n "${server_pid}" ]; then
        kill -- "-${server_pid}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        server_pid=
    fi
}

run_arm() {
    local k=$1
    local width=$((k + 1))
    local log="${RESULTS}/server_k${k}.log"
    echo "=== SGLang NEXTN K=${k}, verify width=${width} ==="

    setsid env \
        FLASHINFER_DISABLE_VERSION_CHECK=1 \
        NCCL_P2P_LEVEL=NVL \
        SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
        SGLANG_MAMBA_CONV_DTYPE=float16 \
        SGLANG_MAMBA_SSM_DTYPE=float16 \
        SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
        sglang serve \
        --trust-remote-code \
        --model-path "${MODEL_DIR}" \
        --dtype float16 \
        --kv-cache-dtype fp8_e5m2 \
        --attention-backend flash_attn_v100 \
        --linear-attn-prefill-backend triton \
        --linear-attn-decode-backend triton \
        --tensor-parallel-size "${TP}" \
        --host 0.0.0.0 \
        --port 8082 \
        --mem-fraction-static 0.75 \
        --context-length 16384 \
        --max-total-tokens 16384 \
        --max-running-requests 1 \
        --disable-radix-cache \
        --chunked-prefill-size 8192 \
        --mamba-full-memory-ratio 0.1 \
        --mamba-scheduler-strategy no_buffer \
        --cuda-graph-max-bs 1 \
        --cuda-graph-bs 1 \
        --speculative-algorithm NEXTN \
        --speculative-num-steps "${k}" \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens "${width}" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        >"${log}" 2>&1 &
    server_pid=$!
    wait_for_server "${log}" || return 1

    python3 /job/sglang_nextn_client.py \
        --base-url http://127.0.0.1:8082 \
        --model "${MODEL_DIR}" \
        --input-tokens 1024 \
        --output-tokens 256 \
        --trials 2 \
        --json-out "${RESULTS}/k${k}.json" || return $?

    curl -fsS http://127.0.0.1:8082/server_info >"${RESULTS}/server_info_k${k}.json" || true
    grep -aE 'avg_spec_accept_length|Decode batch|Prefill batch|throughput' "${log}" | tail -100 || true
    stop_server
}

run_arm 1 || exit $?
run_arm 4 || exit $?

python3 - "${RESULTS}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
print("=== SGLang native NEXTN control ===")
for k in (1, 4):
    row = json.load(open(os.path.join(root, f"k{k}.json")))
    print(
        f"K={k}: decode={row['mean_decode_tok_s']:.2f} tok/s "
        f"accept_length={row['avg_spec_accept_length']}"
    )
PY

touch "${RESULTS}/completed"
echo SGLANG_NEXTN_CONTROL_COMPLETE
