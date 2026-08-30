#!/usr/bin/env bash
set -euo pipefail
export SGLANG_DFLASH_PARITY_BLOCK_INDEX=2
export LM_DFLASH_PARITY_REF=/results/20260830_235055-dflash-second-block-1bc2408cfea5/parity/lmdeploy
exec bash /src/tools/v100/sglang_dflash_parity_job.sh
