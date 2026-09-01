#!/usr/bin/env bash
# Isolate native draft boundaries after exact SGLang context and initial RMSNorm alignment.
set -euo pipefail
export TM_DFLASH_FULL_PRODUCT_RMSNORM=1
exec bash /job/dflash_context_replay_job.sh
