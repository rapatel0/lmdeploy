#!/usr/bin/env bash
# Test audited identity with checkpoint-order QKV weights and cuBLAS GEMMs.
set -euo pipefail
export TM_DFLASH_QKV_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_graph_identity_job.sh
