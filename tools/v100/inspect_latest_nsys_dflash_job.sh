#!/usr/bin/env bash
# Print bounded current-default Nsight summaries from the latest DFlash profile.
set -euo pipefail
ROOT="$(find /results -maxdepth 1 -type d -name '*-nsys-dflash-*' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${ROOT}" ] || { echo 'FAIL: no DFlash Nsight result' >&2; exit 2; }
echo "NSYS_DFLASH_RESULT=${ROOT}"
for suffix in nvtx_sum cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum; do
    file="${ROOT}/k7_stats_${suffix}.csv"
    [ -f "${file}" ] || { echo "FAIL: missing ${file}" >&2; exit 3; }
    echo "=== ${suffix} ==="
    head -40 "${file}"
done
