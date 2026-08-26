#!/usr/bin/env bash
# Build the wheel from the staged source and install it. No model load.
#
# Used to check that a C++ change compiles before spending a model load on it.
# The MTP verification lives in separate drivers.
set -uo pipefail

echo "=== source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

echo
echo "=== build ==="
cd /src
bash /src/tools/v100/build_v100.sh
RC=$?
if [ "${RC}" -ne 0 ]; then
    echo "FAIL: build exited ${RC}" >&2
    exit "${RC}"
fi

echo
echo "=== install ==="
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

echo
echo "=== import check ==="
cd /
python3 -c "
import lmdeploy.turbomind.turbomind  # puts lmdeploy/lib on sys.path
import _turbomind as t
print('  MTPLayerConfig:', hasattr(t, 'MTPLayerConfig'))
" || {
    echo "FAIL: import check failed" >&2
    exit 4
}

echo
echo "BUILD_ONLY_PASS"
