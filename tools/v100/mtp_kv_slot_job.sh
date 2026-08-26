#!/usr/bin/env bash
# Build, install, then prove the MTP draft layer owns a KV slot.
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

echo
echo "=== verify the KV slot ==="
cd /
# Same checkpoint the load and benchmark jobs use. Fail loudly rather than
# falling back, so a missing mount is never mistaken for a code failure.
MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
if [ ! -f "${MODEL}/config.json" ]; then
    echo "FAIL: no checkpoint at ${MODEL}; available:" >&2
    ls -1 /models 2>/dev/null | sed 's/^/  /' >&2
    exit 3
fi

python3 /src/tools/v100/verify_mtp_kv_slot.py --model-dir "${MODEL}" --tp 4
