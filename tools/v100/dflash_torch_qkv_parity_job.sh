#!/usr/bin/env bash
set -euo pipefail
export REPLAY_EXACT_QK=0
export REPLAY_PROJECTED_QK=0
export REPLAY_ATTENTION_INPUT=1
export TM_DFLASH_QK_NORM_WARP32=1
export TM_DFLASH_QK_FULL_PRODUCT=1
export TM_DFLASH_QKV_TORCH_LAYOUT=1
export TM_DFLASH_RANK_ORDERED_ALLREDUCE=${TM_DFLASH_RANK_ORDERED_ALLREDUCE:-1}
exec bash /job/dflash_context_kv_parity_job.sh
