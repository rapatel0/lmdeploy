#!/usr/bin/env bash
# Test Triton-compatible exp2 sigmoid at the Laguna activation boundary.
set -euo pipefail
export TM_DFLASH_LAGUNA_TRITON_SIGMOID=1
exec bash /job/dflash_native_attention_trace_job.sh
