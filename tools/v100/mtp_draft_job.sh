#!/usr/bin/env bash
# Prove the MTP draft path executes. Assumes the wheel is already built.
set -uo pipefail

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "${WHEEL}" ]; then
    echo "FAIL: no wheel in /wheels; run the build job first" >&2
    exit 2
fi
echo "=== installing $(basename "${WHEEL}") ==="
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
[ -f "${MODEL}/config.json" ] || { echo "FAIL: no checkpoint at ${MODEL}" >&2; exit 3; }

echo
echo "=== run the draft path ==="
cd /
python3 /src/tools/v100/verify_mtp_draft.py --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4
