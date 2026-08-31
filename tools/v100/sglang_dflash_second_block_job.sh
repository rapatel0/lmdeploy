#!/usr/bin/env bash
set -euo pipefail
export SGLANG_DFLASH_PARITY_BLOCK_INDEX=2
export LM_DFLASH_PARITY_REF=/results/20260830_234355-dflash-second-block-bb75218d702b/parity/lmdeploy
exec bash /src/tools/v100/sglang_dflash_parity_job.sh
