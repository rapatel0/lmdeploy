#!/usr/bin/env bash
# Qualify fused E4M3 decode and fused decode plus K128 scale retention.
# Pause gpu-01 DCGM before launch so Nsight Compute owns performance counters.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-fused-decode-${SRC_COMMIT}
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

python3 - "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    "${RESULTS}/scale_audit.json" <<'PY'
import glob, json, sys
import torch
from safetensors import safe_open
roots, output = sys.argv[1:-1], sys.argv[-1]
records = []
coverage = {}
for root in roots:
    paths = sorted(glob.glob(root + "/*.safetensors"))
    if not paths:
        raise SystemExit(f"FAIL: no safetensors files found under {root}")
    root_scales = 0
    fp8_weights = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as src:
            for key in src.keys():
                dtype = str(src.get_slice(key).get_dtype()).upper()
                if "F8" in dtype and "weight" in key:
                    fp8_weights += 1
                if "weight_scale" not in key:
                    continue
                root_scales += 1
                value = src.get_tensor(key).float().reshape(-1)
                if not value.numel():
                    continue
                half = value.half()
                bits = half.view(torch.int16).to(torch.int32) & 0xffff
                exponent = (bits >> 10) & 31
                nonzero = half != 0
                bad = nonzero & ((exponent == 0) | (exponent > 22) | ~torch.isfinite(half))
                records.append({
                    "root": root, "file": path.rsplit("/", 1)[-1], "key": key, "count": value.numel(),
                    "min": value.min().item(), "max": value.max().item(),
                    "half_min_exponent": exponent[nonzero].min().item() if nonzero.any() else 0,
                    "half_max_exponent": exponent[nonzero].max().item() if nonzero.any() else 0,
                    "unsafe": bad.sum().item(),
                })
    coverage[root] = {"files": len(paths), "fp8_weight_tensors": fp8_weights, "scale_tensors": root_scales}
    if fp8_weights and not root_scales:
        raise SystemExit(f"FAIL: {root} has FP8 weights but no weight scales")
if not records:
    raise SystemExit("FAIL: no weight_scale tensors found across either checkpoint")
unsafe = sum(row["unsafe"] for row in records)
result = {"tensor_count": len(records), "unsafe_values": unsafe, "coverage": coverage, "records": records}
open(output, "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print(f"SM70_FP8_M8_SCALE_AUDIT tensors={len(records)} unsafe={unsafe} coverage={coverage}")
if unsafe:
    raise SystemExit("FAIL: scale*256 is not safe for every checkpoint scale")
PY

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
    echo "FAIL: wheel predates current build: ${WHEEL}" >&2
    exit 2
}
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

NCU="$(command -v ncu 2>/dev/null || true)"
NVCC="$(command -v nvcc 2>/dev/null || true)"
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then NSYS=/opt/nsys/nsys; fi
[ -n "${NCU}" ] || {
    echo 'FAIL: ncu unavailable' >&2
    exit 2
}
[ -n "${NVCC}" ] || {
    echo 'FAIL: nvcc unavailable' >&2
    exit 2
}
[ -n "${NSYS}" ] || {
    echo 'FAIL: nsys unavailable' >&2
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
    echo 'FAIL: fmt build header unavailable' >&2
    exit 2
}
FMT_INCLUDE="${FMT_HEADER%/fmt/format.h}"
PYLIB="$(find /usr /opt -name 'libpython3.12.so*' -type f -print -quit 2>/dev/null)"
[ -n "${PYLIB}" ] || {
    echo 'FAIL: shared Python library unavailable' >&2
    exit 2
}
"${NVCC}" -std=c++17 -O2 -lineinfo -gencode arch=compute_70,code=sm_70 \
    -I/src -I"${FMT_INCLUDE}" /job/dflash_fp8_m8_microbench.cu "${LIB}" \
    -Xlinker -rpath -Xlinker "${LIBDIR}" -lcudart -Xlinker "${PYLIB}" \
    -o /tmp/dflash_fp8_m8_microbench >"${RESULTS}/compile_microbench.log" 2>&1 || {
    tail -100 "${RESULTS}/compile_microbench.log"
    exit 2
}

arms=(baseline fast combined)
modes=(0 1 2)
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    mkdir -p "${RESULTS}/micro_${arm}"
    TM_SM70_FP8_M8_FUSED_DECODE="${mode}" \
        TM_GEMM_TUNE='top_k=0,clusters=0,max_splits=1,min_iter=2,max_iter=10,max_time=2.0' \
        TM_GEMM_TUNE_VERBOSE=1 TM_FP8_M8_DUMP_DIR="${RESULTS}/micro_${arm}" \
        /tmp/dflash_fp8_m8_microbench 2>&1 | tee "${RESULTS}/micro_${arm}.log"
    grep -q '^SM70_FP8_M8_FUSED_DECODE_PARITY finite=0 ' "${RESULTS}/micro_${arm}.log"
done
for arm in fast combined; do
    for shape in gate_up down out; do
        cmp "${RESULTS}/micro_baseline/${shape}.bin" "${RESULTS}/micro_${arm}/${shape}.bin" || {
            echo "FAIL: ${arm} bitwise mismatch for ${shape}" >&2
            exit 3
        }
    done
done
echo SM70_FP8_M8_FUSED_DECODE_BITWISE_PASS

export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
export TM_GEMM_TUNE_VERBOSE=1
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    TM_SM70_FP8_M8_FUSED_DECODE="${mode}" "${NCU}" \
        --force-overwrite --target-processes all --profile-from-start off \
        --kernel-name-base demangled \
        --kernel-name 'regex:gemm_kernel.*MMA_Map.*int.8.*Operand_B_Pack.*e4m3' \
        --launch-skip 0 --launch-count 3 \
        --section LaunchStats --section Occupancy --section SpeedOfLight \
        --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
        --export "${RESULTS}/ncu_${arm}" /tmp/dflash_fp8_m8_microbench \
        2>&1 | tee "${RESULTS}/ncu_${arm}.log"
    "${NCU}" --import "${RESULTS}/ncu_${arm}.ncu-rep" --csv --page details \
        >"${RESULTS}/ncu_${arm}_details.csv" 2>"${RESULTS}/ncu_${arm}_import.log"
    grep -q 'GPU Speed Of Light Throughput' "${RESULTS}/ncu_${arm}_details.csv"
done
python3 - "${RESULTS}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
wanted = {"Duration", "Registers Per Thread", "Dynamic Shared Memory Per Block", "Theoretical Occupancy",
          "Achieved Occupancy", "Eligible Warps Per Scheduler", "Issued Warp Per Scheduler",
          "Compute (SM) Throughput", "L1/TEX Cache Throughput", "DRAM Throughput"}
summary = {}
for arm in ("baseline", "fast", "combined"):
    records = {}
    with (root / f"ncu_{arm}_details.csv").open(newline="", encoding="utf-8") as src:
        for row in csv.DictReader(src):
            idx = int(row["ID"])
            rec = records.setdefault(idx, {"grid": row["Grid Size"], "block": row["Block Size"]})
            if row["Metric Name"] in wanted:
                value = row["Metric Value"].strip().replace(",", "")
                try:
                    float(value)
                except ValueError as exc:
                    raise SystemExit(f"{arm} launch {idx}: nonnumeric {row['Metric Name']}={value!r}") from exc
                rec[row["Metric Name"]] = {"value": value, "unit": row["Metric Unit"]}
    if set(records) != {0, 1, 2}:
        raise SystemExit(f"{arm}: expected three NCU launches, got {sorted(records)}")
    for idx, rec in records.items():
        missing = wanted - rec.keys()
        if missing or not rec["grid"] or not rec["block"]:
            raise SystemExit(f"{arm} launch {idx}: incomplete NCU report, missing={sorted(missing)}")
    summary[arm] = records
(root / "ncu_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("SM70_FP8_M8_FUSED_DECODE_NCU_PASS")
PY

cd /
export TM_LOG_LEVEL=INFO
unset TM_GEMM_TUNE_VERBOSE
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
    local name=$1 mode=$2
    TM_SM70_FP8_M8_FUSED_DECODE="${mode}" TM_GEMM_DEBUG_OCCUPANCY=1 \
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
for i in "${!arms[@]}"; do run_arm "${arms[$i]}" "${modes[$i]}"; done
for mode in 1 2; do
    log=fast
    [ "${mode}" = 2 ] && log=combined
    for ((device = 0; device < "${TP:-4}"; ++device)); do
        grep -q "SM70_FP8_M8_FUSED_DECODE_REGISTERED device=${device} mode=${mode}" "${RESULTS}/${log}.log" || {
            echo "FAIL: missing mode ${mode} registration for device ${device}" >&2
            exit 3
        }
    done
done

export FT_NVTX=ON
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    TM_SM70_FP8_M8_FUSED_DECODE="${mode}" "${NSYS}" profile \
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
    grep -q 'MMA_884_GROUPED' "${RESULTS}/profile_${arm}_stats_cuda_gpu_kern_sum.csv"
done
unset FT_NVTX
python3 /src/tools/v100/analyze_dflash_fp8_m8_fused_decode.py "${RESULTS}" | tee "${RESULTS}/analysis.log"

echo '=== combined exact audited identity ==='
set +e
TM_SM70_FP8_M8_FUSED_DECODE=2 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
identity_rc=${PIPESTATUS[0]}
set -e
identity_pass=0
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log" && identity_pass=1

set +e
python3 - "${RESULTS}/analysis.json" "${RESULTS}/qualification.json" "${identity_pass}" "${identity_rc}" <<'PY'
import json, sys
analysis = json.load(open(sys.argv[1], encoding="utf-8"))
identity = bool(int(sys.argv[3])) and int(sys.argv[4]) == 0
change = analysis["changes"]["combined"]
checks = {
    "identity": identity,
    "aggregate_m8_improvement_at_least_5pct": change["profile_m8_fp8_pct"] <= -5.0,
    "unprofiled_cycle_improvement_at_least_1pct": change["unprofiled_cycle_pct"] <= -1.0,
    "profiled_cycle_no_regression": change["profiled_cycle_pct"] <= 0.5,
}
result = {"status": "qualified" if all(checks.values()) else "rejected", "checks": checks,
          "changes": analysis["changes"]}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_FP8_M8_FUSED_DECODE_" + result["status"].upper(), json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["status"] == "qualified" else 4)
PY
qualification_rc=$?
set -e

touch "${RESULTS}/completed"
if [ "${qualification_rc}" -ne 0 ]; then
    echo DFLASH_FP8_M8_FUSED_DECODE_REJECTED
    exit "${qualification_rc}"
fi
echo DFLASH_FP8_M8_FUSED_DECODE_COMPLETE
