#!/usr/bin/env bash
# Attribute dominant FP16 M<=8 GEMMs and test existing native SM70 kernels.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-m8-backend-${SRC_COMMIT}
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
        echo "FAIL: missing ${driver}" >&2
        exit 2
    }
done
cat /src/SOURCE_STAMP

rm -f /wheels/lmdeploy-*.whl
build_started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo 'FAIL: current build produced no wheel' >&2
    exit 2
}
[ "$(stat -c %Y "${WHEEL}")" -ge "${build_started}" ] || {
    echo 'FAIL: stale wheel' >&2
    exit 2
}
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then NSYS=/opt/nsys/nsys; fi
[ -n "${NSYS}" ] || {
    echo 'FAIL: nsys unavailable' >&2
    exit 2
}
cd /
export TM_LOG_LEVEL=INFO
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'

validate_bench() {
    local path=$1 trials=$2 tokens=$3
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

bench() {
    local name=$1 trials=$2 tokens=$3 backend=$4 paths=$5 trace=$6
    local -a env_args=(-u TM_GEMM_FP16_M8_BACKEND -u TM_GEMM_FP16_M8_PATHS -u TM_GEMM_TRACE_FP16_M8)
    if [ -n "${backend}" ]; then
        env_args+=("TM_GEMM_FP16_M8_BACKEND=${backend}" "TM_GEMM_FP16_M8_PATHS=${paths}")
    fi
    if [ "${trace}" = 1 ]; then env_args+=(TM_GEMM_TRACE_FP16_M8=1); fi
    env "${env_args[@]}" python3 /job/bench_decode.py \
            --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
            --num-draft-tokens 7 --speculative-algorithm dflash2 \
            --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
            --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
            --input-tokens 1000 --output-tokens "${tokens}" --trials "${trials}" \
            --sglang-corpus /sglang-corpus \
            --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
            --cache-max-entry-count 0.05 --json-out "${RESULTS}/${name}.json" \
        2>&1 | tee "${RESULTS}/${name}.log"
    local bench_rc=${PIPESTATUS[0]}
    if [ "${bench_rc}" -ne 0 ]; then
        return "${bench_rc}"
    fi
    validate_bench "${RESULTS}/${name}.json" "${trials}" "${tokens}"
}

echo '=== route attribution probe ==='
bench route_baseline 1 64 '' '' 1
python3 - "${RESULTS}/route_baseline.log" "${RESULTS}/route_selection.json" <<'PY'
import json, re, sys
pattern = re.compile(r"GEMM_FP16_M8_ROUTE tag=(.*?) m=(\d+) n=(\d+) k=(\d+) backend=(\d+)")
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = pattern.search(line)
    if match:
        tag, m, n, k, backend = match.groups()
        rows.append({"tag": tag, "m": int(m), "n": int(n), "k": int(k), "backend": int(backend)})
head = sorted({row["tag"] for row in rows if row["m"] <= 8 and row["n"] >= 50000 and row["backend"] == 1})
gdn = sorted({row["tag"] for row in rows if "in_proj_all" in row["tag"] and row["m"] <= 8 and row["backend"] == 1})
if len(head) != 1:
    raise SystemExit(f"FAIL: expected one cuBLAS output-head path, got {head}")
if len(gdn) != 48:
    raise SystemExit(f"FAIL: expected 48 cuBLAS GDN in_proj_all paths, got {len(gdn)}")
result = {"head_path": head[0], "gdn_paths": gdn, "routes": rows}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_FP16_M8_ROUTE_ATTRIBUTION_PASS", head[0], len(gdn))
PY
HEAD_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["head_path"])' "${RESULTS}/route_selection.json")"
GDN_PATHS="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["gdn_paths"]))' "${RESULTS}/route_selection.json")"

arms=(baseline native_gdn native_head native_both)
backends=('' native native native)
paths=('' "${GDN_PATHS}" "${HEAD_PATH}" "${GDN_PATHS},${HEAD_PATH}")
valid_arms=(baseline)
bench baseline 5 256 '' '' 0
for i in 1 2 3; do
    arm=${arms[$i]}
    set +e
    bench "${arm}" 5 256 "${backends[$i]}" "${paths[$i]}" 0
    rc=$?
    set -e
    if [ "${rc}" -eq 0 ]; then
        valid_arms+=("${arm}")
        bench "route_${arm}" 1 64 "${backends[$i]}" "${paths[$i]}" 1
        python3 - "${RESULTS}/route_${arm}.log" "${paths[$i]}" <<'PY'
import re, sys
log, selectors = sys.argv[1:]
expected = {token for token in selectors.split(",") if token}
pattern = re.compile(
    r"GEMM_FP16_M8_ROUTE tag=(.*?) m=(\d+) n=(\d+) k=(\d+) backend=(\d+) "
    r"splits=(\d+) swizzle=(\d+) tile=(\d+)x(\d+)x(\d+)"
)
native = set()
wrong = []
for line in open(log, encoding="utf-8", errors="replace"):
    match = pattern.search(line)
    if not match:
        continue
    tag, _, _, _, backend, splits, swizzle, tile_m, tile_n, tile_k = match.groups()
    if tag in expected:
        if int(backend) != 0 or int(tile_m) <= 0 or int(tile_n) <= 0 or int(tile_k) <= 0:
            wrong.append((tag, backend, splits, swizzle, tile_m, tile_n, tile_k))
        else:
            native.add(tag)
    elif int(backend) == 0:
        wrong.append(("unexpected", tag, backend))
if wrong or native != expected:
    missing = sorted(expected - native)
    extra = sorted(native - expected)
    raise SystemExit(f"FAIL: exact forced route mismatch missing={missing} extra={extra} wrong={wrong[:5]}")
print("DFLASH_FP16_M8_FORCED_ROUTE_PASS", len(native))
PY
    else
        echo "DFLASH_FP16_M8_ARM_UNAVAILABLE arm=${arm} rc=${rc}" | tee "${RESULTS}/${arm}.failed"
        rm -f "${RESULTS}/${arm}.json"
    fi
done
if [ "${#valid_arms[@]}" -ne "${#arms[@]}" ]; then
    python3 - "${RESULTS}" "${valid_arms[*]}" <<'PY'
import json, sys
result = {"status": "rejected", "reason": "incomplete four-arm matrix", "valid_arms": sys.argv[2].split()}
open(sys.argv[1] + "/qualification.json", "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_FP16_M8_BACKEND_REJECTED", json.dumps(result, sort_keys=True))
PY
    touch "${RESULTS}/completed"
    exit 4
fi

export FT_NVTX=ON
for i in "${!arms[@]}"; do
    arm=${arms[$i]}
    backend=${backends[$i]} path=${paths[$i]}
    profile_env=(-u TM_GEMM_FP16_M8_BACKEND -u TM_GEMM_FP16_M8_PATHS TM_GEMM_TRACE_FP16_M8=1)
    if [ -n "${backend}" ]; then
        profile_env+=("TM_GEMM_FP16_M8_BACKEND=${backend}" "TM_GEMM_FP16_M8_PATHS=${path}")
    fi
    env "${profile_env[@]}" "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt \
            --cuda-memory-usage=true \
            --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${arm}" \
            python3 /job/bench_decode.py \
            --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
            --num-draft-tokens 7 --speculative-algorithm dflash2 \
            --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
            --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
            --input-tokens 1000 --output-tokens 128 --trials 1 \
            --sglang-corpus /sglang-corpus \
            --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
            --cache-max-entry-count 0.05 --cuda-profiler-range --json-out "${RESULTS}/profile_${arm}.json" \
        2>&1 | tee "${RESULTS}/profile_${arm}.log"
    validate_bench "${RESULTS}/profile_${arm}.json" 1 128
    "${NSYS}" stats --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv --output "${RESULTS}/profile_${arm}_stats" "${RESULTS}/profile_${arm}.nsys-rep" \
        >"${RESULTS}/profile_${arm}_stats.log" 2>&1
done
unset FT_NVTX
python3 /src/tools/v100/analyze_dflash_fp16_m8_backend.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
WINNER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "${RESULTS}/analysis.json")"

set +e
python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_baseline.log"
baseline_identity_rc=${PIPESTATUS[0]}
set -e

winner_identity_rc=4
if [ -n "${WINNER}" ]; then
    for i in 1 2 3; do [ "${arms[$i]}" = "${WINNER}" ] && winner_i=$i; done
    set +e
    TM_GEMM_FP16_M8_BACKEND=native TM_GEMM_FP16_M8_PATHS="${paths[$winner_i]}" \
        python3 /job/verify_dflash_audited.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
        --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        2>&1 | tee "${RESULTS}/identity_winner.log"
    winner_identity_rc=${PIPESTATUS[0]}
    set -e
fi

python3 - "${RESULTS}" "${WINNER}" "${baseline_identity_rc}" "${winner_identity_rc}" <<'PY'
import ast, json, re, sys
root, winner, baseline_rc, winner_rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
analysis = json.load(open(root + "/analysis.json", encoding="utf-8"))
def ids(path, label):
    text = open(path, encoding="utf-8", errors="replace").read()
    match = re.search(rf"^{label} token_ids=(\[.*\])$", text, re.M)
    return ast.literal_eval(match.group(1)) if match else None
baseline_k0 = ids(root + "/identity_baseline.log", "K=0")
identity = False
if winner:
    winner_k0 = ids(root + "/identity_winner.log", "K=0")
    winner_k7 = ids(root + "/identity_winner.log", "K=7")
    identity = (baseline_rc == 0 and winner_rc == 0 and baseline_k0 is not None
                and baseline_k0 == winner_k0 == winner_k7)
result = {"status": "qualified" if winner and identity else "rejected", "winner": winner or None,
          "identity": identity, "changes": analysis["changes"]}
open(root + "/qualification.json", "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_FP16_M8_BACKEND_" + result["status"].upper(), json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["status"] == "qualified" else 4)
PY
qualification_rc=$?
set -e

touch "${RESULTS}/completed"
if [ "${qualification_rc}" -ne 0 ]; then
    echo DFLASH_FP16_M8_BACKEND_REJECTED
    exit "${qualification_rc}"
fi
echo DFLASH_FP16_M8_BACKEND_COMPLETE
