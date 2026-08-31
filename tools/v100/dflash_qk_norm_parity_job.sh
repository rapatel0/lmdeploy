#!/usr/bin/env bash
set -euo pipefail
export REPLAY_EXACT_QK=0
export TM_DFLASH_QK_NORM_WARP32=1
exec /job/dflash_context_kv_parity_job.sh
