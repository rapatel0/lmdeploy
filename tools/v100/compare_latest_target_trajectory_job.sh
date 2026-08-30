#!/usr/bin/env bash
# Re-run target trajectory analysis against the latest retained TP4 traces.
set -euo pipefail
LM_REF="$(find /results -maxdepth 3 -type d -path '*-dflash-target-trajectory-*/parity/lmdeploy' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
SG_REF="$(find /results -maxdepth 3 -type d -path '*-sglang-dflash-parity-*/trace/sglang' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${LM_REF}" ] && [ -n "${SG_REF}" ] || {
    echo "FAIL: missing LMDeploy or SGLang target trace" >&2
    exit 2
}
echo "LM_TARGET_REF=${LM_REF}"
echo "SGLANG_TARGET_REF=${SG_REF}"
python3 /job/compare_dflash_target_trajectory.py \
    --lmdeploy "${LM_REF}" --sglang "${SG_REF}" --output /results/latest-target-trajectory.json
