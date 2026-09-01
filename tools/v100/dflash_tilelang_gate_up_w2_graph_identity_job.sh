#!/usr/bin/env bash
# Test Torch-compatible gate/up and down projections while retaining native attention.
set -euo pipefail
export TM_DFLASH_GATE_UP_TORCH_LAYOUT=1
export TM_DFLASH_W2_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_graph_identity_job.sh
