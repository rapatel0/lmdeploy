#!/usr/bin/env bash
# Test grouped K128 scale reuse in TurboMind's SM70 M=8 block-FP8 kernel.
# Pause gpu-01 DCGM first (`profiling-paused=true`) so NCU owns counters.
# Launch with:
#   tools/v100/run_island2.sh tools/v100/dflash_fp8_m8_reuse_scale_job.sh \
#     lmdeploy-dflash-fp8-m8-reuse tools/v100/bench_decode.py \
#     tools/v100/verify_dflash_audited.py tools/v100/dflash_fp8_m8_microbench.cu
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-reuse-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

for driver in /job/bench_decode.py /job/verify_dflash_audited.py /job/dflash_fp8_m8_microbench.cu; do
    [ -f "${driver}" ] || {
        echo "FAIL: missing ${driver}" >&2
        exit 2
    }
done

cat /src/SOURCE_STAMP
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo "FAIL: no built wheel" >&2
    exit 2
}
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

NCU="$(command -v ncu 2>/dev/null || true)"
NVCC="$(command -v nvcc 2>/dev/null || true)"
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then
    NSYS=/opt/nsys/nsys
fi
[ -n "${NCU}" ] || {
    echo "FAIL: ncu unavailable" >&2
    exit 2
}
[ -n "${NVCC}" ] || {
    echo "FAIL: nvcc unavailable" >&2
    exit 2
}
[ -n "${NSYS}" ] || {
    echo "FAIL: nsys unavailable" >&2
    exit 2
}

LIB="$(
    python3 - <<'PY'
import glob
paths = glob.glob('/opt/venv/lib/python*/site-packages/lmdeploy/lib/_turbomind*.so')
if not paths:
    raise SystemExit(1)
print(paths[0])
PY
)"
LIBDIR="$(dirname "${LIB}")"
FMT_HEADER="$(find /src/build -path '*/fmt/format.h' -print -quit)"
[ -n "${FMT_HEADER}" ] || {
    echo "FAIL: fmt build header unavailable" >&2
    exit 2
}
FMT_INCLUDE="${FMT_HEADER%/fmt/format.h}"
PYLIB="$(find /usr /opt -name 'libpython3.12.so*' -type f -print -quit 2>/dev/null)"
[ -n "${PYLIB}" ] || {
    echo "FAIL: shared Python library unavailable" >&2
    exit 2
}
"${NVCC}" -std=c++17 -O2 -lineinfo -gencode arch=compute_70,code=sm_70 \
    -I/src -I"${FMT_INCLUDE}" /job/dflash_fp8_m8_microbench.cu "${LIB}" \
    -Xlinker -rpath -Xlinker "${LIBDIR}" -lcudart -Xlinker "${PYLIB}" \
    -o /tmp/dflash_fp8_m8_microbench >"${RESULTS}/compile_microbench.log" 2>&1 || {
    tail -100 "${RESULTS}/compile_microbench.log"
    exit 2
}

# Same-launch arithmetic proof: one split, deterministic nonzero packed inputs,
# and varying physical K-group scales in two fresh processes.
for arm in baseline reuse; do
    reuse=0
    [ "${arm}" = reuse ] && reuse=1
    mkdir -p "${RESULTS}/micro_${arm}"
    TM_SM70_FP8_M8_REUSE_SCALE="${reuse}" \
        TM_GEMM_TUNE='top_k=0,clusters=0,max_splits=1,min_iter=2,max_iter=10,max_time=2.0' \
        TM_GEMM_TUNE_VERBOSE=1 TM_FP8_M8_DUMP_DIR="${RESULTS}/micro_${arm}" \
        /tmp/dflash_fp8_m8_microbench 2>&1 | tee "${RESULTS}/micro_${arm}.log"
done
for shape in gate_up down out; do
    cmp "${RESULTS}/micro_baseline/${shape}.bin" "${RESULTS}/micro_reuse/${shape}.bin" || {
        echo "FAIL: grouped-scale output mismatch for ${shape}" >&2
        exit 3
    }
done
python3 - "${RESULTS}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name in ("gate_up", "down", "out"):
    data = (root / "micro_reuse" / f"{name}.bin").read_bytes()
    if not any(data):
        raise SystemExit(f"{name}: deterministic output is all zero")
print("SM70_FP8_M8_REUSE_SCALE_BITWISE_PASS")
PY

grep -q '^SM70_FP8_M8_REUSE_SCALE_REGISTERED device=0$' "${RESULTS}/micro_reuse.log" || {
    echo "FAIL: single-GPU reuse registration missing" >&2
    exit 3
}

# Production-autotuned NCU comparison. cudaProfilerStart/Stop excludes tuning.
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
export TM_GEMM_TUNE_VERBOSE=1
for arm in baseline reuse; do
    reuse=0
    [ "${arm}" = reuse ] && reuse=1
    TM_SM70_FP8_M8_REUSE_SCALE="${reuse}" "${NCU}" \
        --force-overwrite --target-processes all --profile-from-start off \
        --kernel-name-base demangled \
        --kernel-name 'regex:gemm_kernel.*MMA_Map.*int.8.*Operand_B_Pack.*e4m3' \
        --launch-skip 0 --launch-count 3 \
        --section LaunchStats --section Occupancy --section SpeedOfLight \
        --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
        --export "${RESULTS}/ncu_${arm}" /tmp/dflash_fp8_m8_microbench \
        2>&1 | tee "${RESULTS}/ncu_${arm}.log"
    [ -f "${RESULTS}/ncu_${arm}.ncu-rep" ] || {
        echo "FAIL: ${arm} NCU report missing" >&2
        exit 3
    }
    "${NCU}" --import "${RESULTS}/ncu_${arm}.ncu-rep" --csv --page details \
        >"${RESULTS}/ncu_${arm}_details.csv" 2>"${RESULTS}/ncu_${arm}_import.log"
    grep -q 'GPU Speed Of Light Throughput' "${RESULTS}/ncu_${arm}_details.csv"
done
python3 - "${RESULTS}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
wanted = {
    "Duration", "Registers Per Thread", "Dynamic Shared Memory Per Block",
    "Theoretical Occupancy", "Achieved Occupancy", "Eligible Warps Per Scheduler",
    "Issued Warp Per Scheduler", "Compute (SM) Throughput", "L1/TEX Cache Throughput",
    "DRAM Throughput",
}
summary = {}
for arm in ("baseline", "reuse"):
    records = {}
    with (root / f"ncu_{arm}_details.csv").open(newline="", encoding="utf-8") as src:
        for row in csv.DictReader(src):
            idx = int(row["ID"])
            rec = records.setdefault(idx, {"grid": row["Grid Size"], "block": row["Block Size"]})
            if row["Metric Name"] in wanted:
                rec[row["Metric Name"]] = {"value": row["Metric Value"], "unit": row["Metric Unit"]}
    if set(records) != {0, 1, 2}:
        raise SystemExit(f"{arm}: expected three NCU launches, got {sorted(records)}")
    for idx, rec in records.items():
        if "Duration" not in rec or "Registers Per Thread" not in rec:
            raise SystemExit(f"{arm} launch {idx}: incomplete NCU metrics")
    summary[arm] = records
(root / "ncu_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("SM70_FP8_M8_REUSE_SCALE_NCU_PASS")
PY

cd /
export TM_LOG_LEVEL=INFO
unset TM_DFLASH_GROUPED_PAGED_Q8
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
    local name=$1 reuse=$2
    echo "=== arm=${name} reuse_scale=${reuse} ==="
    TM_SM70_FP8_M8_REUSE_SCALE="${reuse}" TM_GEMM_DEBUG_OCCUPANCY=1 \
        python3 /job/bench_decode.py \
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
run_arm reuse 1

expected_tp="${TP:-4}"
registration_count="$(grep -ao 'SM70_FP8_M8_REUSE_SCALE_REGISTERED device=[0-9]\+' "${RESULTS}/reuse.log" | wc -l)"
[ "${registration_count}" -eq "${expected_tp}" ] || {
    echo "FAIL: expected ${expected_tp} reuse registrations, got ${registration_count}" >&2
    exit 3
}
for ((device = 0; device < expected_tp; ++device)); do
    grep -q "SM70_FP8_M8_REUSE_SCALE_REGISTERED device=${device}" "${RESULTS}/reuse.log" || {
        echo "FAIL: missing reuse registration for device ${device}" >&2
        exit 3
    }
done

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
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
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
echo DFLASH_FP8_M8_REUSE_SCALE_COMPLETE
