#!/usr/bin/env bash
# Qualify the identity-safe gate/up Torch-layout improvement on the long corpus run.
set -euo pipefail
export TM_DFLASH_GATE_UP_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_performance_job.sh
