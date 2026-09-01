#!/usr/bin/env bash
set -euo pipefail
: "${SGLANG_TRACE:=/results/20260901_082555-sglang-dflash-parity-48f21dc99772/trace/sglang}"
: "${LMDEPLOY_TRACE:=/results/20260901_113320-dflash-native-attention-d664266cb5af/parity/lmdeploy}"
python3 /job/analyze_dflash_selector_parity.py --lmdeploy "$LMDEPLOY_TRACE" --sglang "$SGLANG_TRACE"
