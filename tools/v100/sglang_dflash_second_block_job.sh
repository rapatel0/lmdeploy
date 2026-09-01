#!/usr/bin/env bash
set -euo pipefail
export SGLANG_DFLASH_PARITY_BLOCK_INDEX=2
export SGLANG_DFLASH_EXPECTED_ANCHOR=1144
export SGLANG_PARITY_TRACE_ONLY=1
exec bash /job/sglang_dflash_parity_job.sh
