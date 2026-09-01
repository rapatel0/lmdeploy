#!/usr/bin/env bash
set -euo pipefail
LM=/results/20260901_113320-dflash-native-attention-d664266cb5af/parity/lmdeploy
SG=/results/20260901_120004-sglang-dflash-parity-9676b0c75cf8/trace/sglang
OUT=/results/dflash-intermediate-parity-$(date +%Y%m%d_%H%M%S).json
python3 /job/compare_dflash_parity.py --lmdeploy "$LM" --sglang "$SG" --output "$OUT"
