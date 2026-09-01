#!/usr/bin/env bash
# Test SGLang-compatible cuBLAS only for the DFlash target-context projector.
set -euo pipefail
export TM_DFLASH_CONTEXT_FC_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_graph_identity_job.sh
