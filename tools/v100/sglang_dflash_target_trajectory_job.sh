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
# The live ModelRunner probe established that the merged 24-row target
# verification slab's final row is logical position 1007. TurboMind likewise
# retains the final verifier row. input_ids is absent at this boundary, so the
# separately audited checkpoint-token-1596 embedding remains the token proof.
export SGLANG_DFLASH_TARGET_POSITION=1007
export SGLANG_DFLASH_TARGET_TOKEN_ID=-1
# Target boundary hooks need the live eager prefill, not a compiled replay.
# The generic parity job retains production CUDA graphs by default.
export SGLANG_PARITY_DISABLE_CUDA_GRAPH=1
exec bash /src/tools/v100/sglang_dflash_parity_job.sh
