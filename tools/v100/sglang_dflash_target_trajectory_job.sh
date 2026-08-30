#!/usr/bin/env bash
# Compare the read-only SGLang target trajectory against the latest TurboMind trace.
set -euo pipefail
LM_REF="$(find /results -maxdepth 3 -type d -path '*-dflash-target-trajectory-*/parity/lmdeploy' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${LM_REF}" ] || {
    echo "FAIL: no TurboMind target trajectory reference exists under /results" >&2
    exit 4
}
export LM_DFLASH_PARITY_REF="${LM_REF}"
export LM_DFLASH_TARGET_TRAJECTORY_REF="${LM_REF}"
exec bash /src/tools/v100/sglang_dflash_parity_job.sh
