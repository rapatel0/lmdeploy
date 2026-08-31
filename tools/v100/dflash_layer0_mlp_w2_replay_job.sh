#!/usr/bin/env bash
set -euo pipefail
export TM_DFLASH_LAYER0_MLP_W2_REPLAY_FILE=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang/rank-0-pid-208/000052-layer0.mlp.pre_conv_side1.bin
exec bash /job/dflash_layer0_mlp_conv_replay_job.sh
