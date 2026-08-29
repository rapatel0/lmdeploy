#!/usr/bin/env bash
# Resume the scale-reuse qualification after a completed build/micro/NCU/A-B run.
# Launch with bench_decode.py and verify_dflash_audited.py as extra files.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
SOURCE_RESULTS="$(find /results -maxdepth 1 -type d -name '*-dflash-fp8-m8-reuse-*' ! -name '*-followup-*' -print | sort | tail -1)"
[ -n "${SOURCE_RESULTS}" ] || {
    echo 'FAIL: no prior scale-reuse result' >&2
    exit 2
}
for file in baseline.json baseline.log reuse.json reuse.log ncu_summary.json wheel.sha256; do
    [ -f "${SOURCE_RESULTS}/${file}" ] || {
        echo "FAIL: missing ${SOURCE_RESULTS}/${file}" >&2
        exit 2
    }
done
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-reuse-followup-${SRC_COMMIT}
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
printf 'resuming from %s\n' "${SOURCE_RESULTS}"
cp "${SOURCE_RESULTS}"/{baseline.json,baseline.log,reuse.json,reuse.log,ncu_summary.json,wheel.sha256} "${RESULTS}/"

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo 'FAIL: no built wheel' >&2
    exit 2
}
expected_sha="$(cut -d' ' -f1 "${SOURCE_RESULTS}/wheel.sha256")"
actual_sha="$(sha256sum "${WHEEL}" | cut -d' ' -f1)"
[ "${actual_sha}" = "${expected_sha}" ] || {
    echo "FAIL: wheel SHA ${actual_sha} does not match prior run ${expected_sha}" >&2
    exit 2
}
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then
    NSYS=/opt/nsys/nsys
fi
[ -n "${NSYS}" ] || {
    echo 'FAIL: nsys unavailable' >&2
    exit 2
}

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
validate_bench "${RESULTS}/baseline.json" 5 256
validate_bench "${RESULTS}/reuse.json" 5 256
for ((device = 0; device < "${TP:-4}"; ++device)); do
    grep -q "SM70_FP8_M8_REUSE_SCALE_REGISTERED device=${device}" "${RESULTS}/reuse.log" || {
        echo "FAIL: missing reuse registration for device ${device}" >&2
        exit 3
    }
done

cd /
export TM_LOG_LEVEL=INFO
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
export TM_GEMM_TUNE_VERBOSE=1
export FT_NVTX=ON
for arm in baseline reuse; do
    reuse=0
    [ "${arm}" = reuse ] && reuse=1
    TM_SM70_FP8_M8_REUSE_SCALE="${reuse}" "${NSYS}" profile \
        --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="${RESULTS}/profile_${arm}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 128 --trials 1 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --cuda-profiler-range \
        --json-out "${RESULTS}/profile_${arm}.json" \
        2>&1 | tee "${RESULTS}/profile_${arm}.log"
    validate_bench "${RESULTS}/profile_${arm}.json" 1 128
    "${NSYS}" stats \
        --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
        --format csv --output "${RESULTS}/profile_${arm}_stats" \
        "${RESULTS}/profile_${arm}.nsys-rep" >"${RESULTS}/profile_${arm}_stats.log" 2>&1
    grep -q 'MMA_884_GROUPED' "${RESULTS}/profile_${arm}_stats_cuda_gpu_kern_sum.csv" || {
        echo "FAIL: grouped Q8H4 attention missing from ${arm} profile" >&2
        exit 3
    }
done
unset FT_NVTX

python3 /src/tools/v100/analyze_dflash_fp8_m8_reuse_scale.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
grep -q '^DFLASH_FP8_M8_REUSE_SCALE_ANALYSIS ' "${RESULTS}/analysis.log"

echo '=== reuse-scale exact audited identity ==='
TM_SM70_FP8_M8_REUSE_SCALE=1 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 |
    tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

python3 - "${RESULTS}/analysis.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "M8 kernel improvement below 1%": data["profile_m8_fp8_change_pct"] <= -1.0,
    "unprofiled cycle improvement below 0.5%": data["unprofiled_cycle_change_pct"] <= -0.5,
    "profiled cycle regression above 0.5%": data["profiled_cycle_change_pct"] <= 0.5,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("DFLASH_FP8_M8_REUSE_SCALE_PERFORMANCE_FAIL: " + "; ".join(failed))
print("DFLASH_FP8_M8_REUSE_SCALE_PERFORMANCE_PASS")
PY

touch "${RESULTS}/completed"
echo DFLASH_FP8_M8_REUSE_SCALE_FOLLOWUP_COMPLETE
