#!/usr/bin/env bash
# Isolate Laguna activation semantics by replaying exact SGLang layer-0 gate/up.
set -euo pipefail
export TM_DFLASH_NATIVE_REPLAY_GATE_UP=1
exec bash /job/dflash_native_attention_trace_job.sh
