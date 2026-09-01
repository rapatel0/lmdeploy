#!/usr/bin/env bash
# Test Torch-compatible attention QKV plus gate/up, retaining native Wo/W2/context/selector.
set -euo pipefail
export TM_DFLASH_ATTENTION_QKV_TORCH_LAYOUT=1
export TM_DFLASH_GATE_UP_TORCH_LAYOUT=1
exec bash /job/dflash_tilelang_graph_identity_job.sh
