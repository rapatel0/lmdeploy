#!/usr/bin/env bash
# Measure target-feature and logit parity with Torch-compatible W2 accumulation.
set -euo pipefail
export TM_DFLASH_W2_TORCH_LAYOUT=1
exec bash /job/dflash_target_trajectory_job.sh
