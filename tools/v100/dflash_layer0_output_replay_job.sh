#!/usr/bin/env bash
set -euo pipefail
export TM_DFLASH_DRAFT_ATTENTION_OUTPUT_REPLAY_FILE=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang/rank-0-pid-208/000044-layer0.attention.wo_reduced.bin
exec bash /job/dflash_torch_qkv_parity_job.sh
