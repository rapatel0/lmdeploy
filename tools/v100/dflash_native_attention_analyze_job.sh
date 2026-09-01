#!/usr/bin/env bash
# Reanalyze durable native attention traces without rebuilding or rerunning a model.
set -euo pipefail
LM=${LM_DFLASH_NATIVE_TRACE_ROOT:-/results/20260901_102031-dflash-native-attention-79ef83d24209/parity/lmdeploy}
SG=${SGLANG_DFLASH_TRACE_ROOT:-/results/20260901_082555-sglang-dflash-parity-48f21dc99772/trace/sglang}
OUT=${DFLASH_NATIVE_ANALYSIS_OUT:-/results/20260901_102031-dflash-native-attention-79ef83d24209}
python3 /job/validate_sglang_dflash_trace.py "$SG" --block-index 1
python3 /job/compare_dflash_attention_layers.py \
    --lmdeploy "$LM" \
    --sglang "$SG" \
    --output "$OUT/all_layer_attention_v2.json" | tee "$OUT/all_layer_attention_v2.log"
python3 /job/compare_dflash_parity.py \
    --lmdeploy "$LM" \
    --sglang "$SG" \
    --output "$OUT/all_layer_model_parity.json" | tee "$OUT/all_layer_model_parity.log"
touch "$OUT/analysis_v2_completed"
echo DFLASH_NATIVE_ATTENTION_REANALYSIS_COMPLETE
