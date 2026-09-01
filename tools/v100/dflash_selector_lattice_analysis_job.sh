#!/usr/bin/env bash
set -euo pipefail
: "${LMDEPLOY_TRACE:=/results/20260901_113320-dflash-native-attention-d664266cb5af/parity/lmdeploy}"
if [ -z "${SGLANG_TRACE:-}" ]; then
  SGLANG_TRACE=$(find /results -maxdepth 3 -type d -path '*-sglang-dflash-parity-e0a4792d*/trace/sglang' | sort | tail -1)
fi
[ -d "$SGLANG_TRACE" ]
python3 /job/analyze_dflash_selector_lattice.py --lmdeploy "$LMDEPLOY_TRACE" --sglang "$SGLANG_TRACE"
