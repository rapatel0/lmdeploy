#!/usr/bin/env bash
set -euo pipefail
BASE=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang/rank-0-pid-208
export TM_DFLASH_LAYER0_RESIDUAL_REPLAY_FILE=${BASE}/000055-layer0.output.residual.bin
export TM_DFLASH_LAYER0_MLP_OUTPUT_REPLAY_FILE=${BASE}/000053-layer0.mlp.conv_side1.bin
exec bash /job/dflash_layer0_mlp_w2_replay_job.sh
