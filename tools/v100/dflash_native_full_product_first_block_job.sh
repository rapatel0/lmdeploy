#!/usr/bin/env bash
# Compare correlated native blocks after exact SGLang-style initial RMSNorm.
set -euo pipefail
export TM_DFLASH_FULL_PRODUCT_RMSNORM=1
export TM_DFLASH_PARITY_BLOCK_INDEX=2
exec bash /job/dflash_native_first_block_parity_job.sh
