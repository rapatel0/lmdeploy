#!/usr/bin/env bash
# Determine whether the earliest target mismatch is embedding load or trace alignment.
set -euo pipefail
LM_REF="$(find /results -maxdepth 3 -type d -path '*-dflash-target-trajectory-*/parity/lmdeploy' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
SG_REF="$(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${LM_REF}" ] || {
    echo 'FAIL: no TurboMind target trajectory' >&2
    exit 4
}
[ -n "${SG_REF}" ] || {
    echo 'FAIL: no SGLang target trajectory' >&2
    exit 4
}
python3 /job/analyze_dflash_target_embedding.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --lmdeploy "${LM_REF}" --sglang "${SG_REF}" --token-id "${TOKEN_ID:-1596}"
