#!/usr/bin/env bash
# Trace gate/up-only Torch-layout parity against SGLang.
set -euo pipefail
export TM_DFLASH_GATE_UP_TORCH_LAYOUT=1
exec bash /job/dflash_native_attention_trace_job.sh
