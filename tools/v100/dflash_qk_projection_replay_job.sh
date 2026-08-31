#!/usr/bin/env bash
set -euo pipefail
export REPLAY_EXACT_QK=0
export REPLAY_PROJECTED_QK=1
export TM_DFLASH_QK_NORM_WARP32=1
export TM_DFLASH_QK_FULL_PRODUCT=1
exec bash /job/dflash_context_kv_parity_job.sh
