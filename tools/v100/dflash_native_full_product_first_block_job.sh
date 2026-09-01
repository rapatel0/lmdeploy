#!/usr/bin/env bash
# Compare correlated native blocks after exact SGLang-style initial RMSNorm.
set -euo pipefail
export TM_DFLASH_FULL_PRODUCT_RMSNORM=1
exec bash /job/dflash_native_first_block_parity_job.sh
