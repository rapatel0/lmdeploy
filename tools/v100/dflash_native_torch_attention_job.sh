#!/usr/bin/env bash
# Diagnose native parity with checkpoint-order QKV weights and cuBLAS Torch-layout GEMMs.
set -euo pipefail
export TM_DFLASH_QKV_TORCH_LAYOUT=1
exec bash /job/dflash_native_attention_trace_job.sh
