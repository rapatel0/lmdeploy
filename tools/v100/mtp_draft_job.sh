#!/usr/bin/env bash
# Build, install, then prove the MTP draft path actually executes.
set -uo pipefail

echo "=== source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

echo
echo "=== build ==="
cd /src
bash /src/tools/v100/build_v100.sh || {
    echo "FAIL: build" >&2
    exit 2
}

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
if [ ! -f "${MODEL}/config.json" ]; then
    echo "FAIL: no checkpoint at ${MODEL}" >&2
    exit 3
fi

echo
echo "=== run the draft path ==="
cd /
python3 /src/tools/v100/verify_mtp_draft.py --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4
