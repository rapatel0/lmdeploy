#!/usr/bin/env bash
# Isolate whether residual context-K/V RoPE rounding causes the sharp layer-0
# attention divergence once Torch-layout QKV projection is exact.
set -euo pipefail
export TM_DFLASH_QKV_TORCH_LAYOUT=1
export TM_DFLASH_NATIVE_REPLAY_FLATTENED=1
exec bash /job/dflash_native_attention_trace_job.sh
