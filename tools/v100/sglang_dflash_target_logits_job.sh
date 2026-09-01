#!/usr/bin/env bash
# Capture one correlated SGLang DFlash block and its preceding target logits.
set -euo pipefail
export SGLANG_PARITY_TRACE_ONLY=1
exec bash /job/sglang_dflash_parity_job.sh
