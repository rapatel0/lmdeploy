#!/usr/bin/env bash
# Qualify SM70 Q=8 GDN value-column CTA decomposition.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-gdn-value-cols-${SRC_COMMIT}
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
arms=(v128 v64 v32 v16)
values=(128 64 32 16)

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

run_arm() {
    local name=$1 value=$2 trials=$3 tokens=$4
    TM_GDN_SM70_VALUE_COLS="${value}" python3 /job/bench_decode.py \
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
    [ "${bench_rc}" -eq 0 ] || return "${bench_rc}"
    validate_bench "${RESULTS}/${name}.json" "${trials}" "${tokens}"
}

for i in "${!arms[@]}"; do
    run_arm "${arms[$i]}" "${values[$i]}" 5 256
    if [ "${values[$i]}" -ne 128 ]; then
        for ((device = 0; device < "${TP:-4}"; ++device)); do
            grep -q "GDN_SM70_VALUE_COLS_ACTIVE device=${device} value_cols=${values[$i]}" \
                "${RESULTS}/${arms[$i]}.log" || {
                echo "FAIL: missing V${values[$i]} activation on device ${device}" >&2
                exit 3
            }
        done
    fi
done

export FT_NVTX=ON
for i in "${!arms[@]}"; do
    arm=${arms[$i]} value=${values[$i]}
    TM_GDN_SM70_VALUE_COLS="${value}" "${NSYS}" profile \
        --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
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
    grep -q "ChunkedGdrKernel<(int)128, (int)16, (int)${value}," \
        "${RESULTS}/profile_${arm}_stats_cuda_gpu_kern_sum.csv"
done
unset FT_NVTX
python3 /src/tools/v100/analyze_dflash_gdn_value_cols.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
WINNER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "${RESULTS}/analysis.json")"

set +e
TM_GDN_SM70_VALUE_COLS=128 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_v128.log"
baseline_identity_rc=${PIPESTATUS[0]}
set -e

winner_identity_rc=4
if [ -n "${WINNER}" ]; then
    winner_value=${WINNER#v}
    set +e
    TM_GDN_SM70_VALUE_COLS="${winner_value}" python3 /job/verify_dflash_audited.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
        --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        2>&1 | tee "${RESULTS}/identity_winner.log"
    winner_identity_rc=${PIPESTATUS[0]}
    set -e
fi

set +e
python3 - "${RESULTS}" "${WINNER}" "${baseline_identity_rc}" "${winner_identity_rc}" <<'PY'
import ast, json, re, sys
root, winner, baseline_rc, winner_rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
analysis = json.load(open(root + "/analysis.json", encoding="utf-8"))
def ids(path, label):
    text = open(path, encoding="utf-8", errors="replace").read()
    match = re.search(rf"^{label} token_ids=(\[.*\])$", text, re.M)
    return ast.literal_eval(match.group(1)) if match else None
baseline_k0 = ids(root + "/identity_v128.log", "K=0")
identity = False
if winner:
    winner_k0 = ids(root + "/identity_winner.log", "K=0")
    winner_k7 = ids(root + "/identity_winner.log", "K=7")
    identity = (baseline_rc == 0 and winner_rc == 0 and baseline_k0 is not None
                and baseline_k0 == winner_k0 == winner_k7)
result = {"status": "qualified" if winner and identity else "rejected", "winner": winner or None,
          "identity": identity, "changes": analysis["changes"], "kernels": analysis["kernels"]}
open(root + "/qualification.json", "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_GDN_VALUE_COLS_" + result["status"].upper(), json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["status"] == "qualified" else 4)
PY
qualification_rc=$?
set -e

touch "${RESULTS}/completed"
if [ "${qualification_rc}" -ne 0 ]; then
    echo DFLASH_GDN_VALUE_COLS_REJECTED
    exit "${qualification_rc}"
fi
echo DFLASH_GDN_VALUE_COLS_COMPLETE
