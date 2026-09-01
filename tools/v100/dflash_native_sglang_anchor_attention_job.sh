#!/usr/bin/env bash
# Trace full model parity with SGLang's input-token draft anchor.
set -euo pipefail
export TM_DFLASH_SGLANG_INPUT_ANCHOR=1
exec bash /job/dflash_native_attention_trace_job.sh
