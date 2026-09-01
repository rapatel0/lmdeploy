#!/usr/bin/env bash
set -euo pipefail
LM=/results/20260901_115102-dflash-native-attention-1fad80449603/parity/lmdeploy
SG=/results/20260901_122037-sglang-dflash-parity-c08cfaef2435/trace/sglang
OUT=/results/dflash-torch-intermediate-parity-$(date +%Y%m%d_%H%M%S).json
python3 /job/compare_dflash_parity.py --lmdeploy "$LM" --sglang "$SG" --output "$OUT"
