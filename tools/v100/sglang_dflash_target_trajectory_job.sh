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
# Compare the same audited prompt boundary in both runtimes. TurboMind's
# independent target trace now persists prompt position 999/token 198 before
# proposal generation instead of piggybacking on a divergent verifier slab.
export SGLANG_DFLASH_TARGET_POSITION=999
export SGLANG_DFLASH_TARGET_TOKEN_ID=198
# Target boundary hooks need the live eager prefill, not a compiled replay.
# The generic parity job retains production CUDA graphs by default.
export SGLANG_PARITY_DISABLE_CUDA_GRAPH=1
exec bash /src/tools/v100/sglang_dflash_parity_job.sh
