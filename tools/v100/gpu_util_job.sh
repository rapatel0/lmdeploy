#!/usr/bin/env bash
# Measure all four visible V100s during matched K=0/K=1/K=4 decode runs.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-gpuutil-${SRC_COMMIT}
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
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    echo "FAIL: build" >&2
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -20
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

sample_gpu() {
    local out=$1
    echo 'epoch_ms,uuid,gpu_util_pct,mem_util_pct,power_w,sm_clock_mhz,memory_mib' >"${out}"
    while :; do
        local now
        now=$(date +%s%3N)
        nvidia-smi \
            --query-gpu=uuid,utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used \
            --format=csv,noheader,nounits |
            awk -v t="${now}" -F', ' '{print t "," $1 "," $2 "," $3 "," $4 "," $5 "," $6}' >>"${out}"
        sleep 0.1
    done
}

run_arm() {
    local k=$1
    local csv="${RESULTS}/gpu_k${k}.csv"
    echo "=== K=${k} ==="
    sample_gpu "${csv}" &
    local sampler=$!
    local args=(
        --model "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens "${k}"
        --input-tokens 1024 --output-tokens 256 --trials 2
        --json-out "${RESULTS}/bench_k${k}.json"
    )
    if [ "${k}" -gt 0 ]; then
        args+=(--require-mtp)
    fi
    python3 /job/bench_decode.py "${args[@]}"
    local rc=$?
    kill "${sampler}" 2>/dev/null || true
    wait "${sampler}" 2>/dev/null || true
    return "${rc}"
}

for k in 0 1 4; do
    run_arm "${k}" || exit $?
done

python3 - "${RESULTS}" <<'PY'
import csv, glob, json, os, statistics, sys
root = sys.argv[1]
print("=== GPU utilization summary ===")
for path in sorted(glob.glob(os.path.join(root, "gpu_k*.csv"))):
    k = os.path.basename(path).split("k", 1)[1].split(".", 1)[0]
    rows = list(csv.DictReader(open(path)))
    by_gpu = {}
    for row in rows:
        by_gpu.setdefault(row["uuid"], []).append(row)
    print(f"K={k}: samples={len(rows)}")
    arm_utils = []
    for uuid, samples in sorted(by_gpu.items()):
        util = [float(x["gpu_util_pct"]) for x in samples]
        power = [float(x["power_w"]) for x in samples]
        clock = [float(x["sm_clock_mhz"]) for x in samples]
        memory = [float(x["memory_mib"]) for x in samples]
        arm_utils.extend(util)
        print(
            f"  {uuid}: avg={statistics.mean(util):.1f}% p50={statistics.median(util):.1f}% "
            f"max={max(util):.0f}% power={statistics.mean(power):.1f}W "
            f"clock={statistics.mean(clock):.0f}MHz memory_max={max(memory):.0f}MiB"
        )
    print(f"  all-GPU average={statistics.mean(arm_utils):.1f}%")
    bench = json.load(open(os.path.join(root, f"bench_k{k}.json")))
    print(f"  decode={bench['mean_decode_tok_s']:.2f} tok/s inclusive={bench['mean_inclusive_tok_s']:.2f} tok/s")
PY

touch "${RESULTS}/completed"
echo GPU_UTIL_COMPLETE
