#!/usr/bin/env bash
# Test SGLang-compatible cuBLAS only for DFlash gate/up projections.
set -euo pipefail
export TM_DFLASH_GATE_UP_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_graph_identity_job.sh
