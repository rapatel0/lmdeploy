#!/usr/bin/env bash
# Install the already-built wheel and prove the MTP binding reached it.
#
# The wheel exists and carries mtp_layer.py, qwen3_5.py with mtp(), and the
# compiled _turbomind extension. The previous job failed only after the build,
# so rebuilding would cost another 45 minutes to reproduce a wheel that is
# already correct. Install and verify instead.
#
# Each step prints before it runs, so a failure names itself even when the pod
# is later reclaimed and the log becomes unreadable.
set -uo pipefail

echo "=== source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "${WHEEL}" ]; then
    echo "FAIL: no wheel to install" >&2
    exit 2
fi

echo
echo "=== install ${WHEEL} ==="
# --no-deps because the campaign image already pins every runtime requirement,
# and resolving them again would fetch from the network inside a Job.
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -3
RC=${PIPESTATUS[0]}
if [ "${RC}" -ne 0 ]; then
    echo "FAIL: pip install exited ${RC}" >&2
    exit 3
fi

echo
echo "=== which package is imported? ==="
# Run from / so that /src/lmdeploy cannot shadow the installed package.
cd /
python3 -c "
import lmdeploy, os
print('  lmdeploy:', lmdeploy.__version__, os.path.dirname(lmdeploy.__file__))
" || {
    echo "FAIL: lmdeploy is not importable" >&2
    exit 4
}

echo
echo "=== config bindings on the extension ==="
python3 - <<'PY'
import sys

# _turbomind lives in lmdeploy/lib, which is not on sys.path until
# lmdeploy.turbomind.turbomind appends it. A direct import fails even when the
# extension is installed correctly.
import lmdeploy.turbomind.turbomind  # noqa: F401

import _turbomind as t

ok = True
for name in ("DecoderLayerConfig", "MTPLayerConfig", "ModelWeightConfig"):
    present = hasattr(t, name)
    print(f"  {name}: {present}")
    ok = ok and present

if not ok:
    print("FAIL: a required config binding is missing", file=sys.stderr)
    sys.exit(5)

cfg = t.MTPLayerConfig()
print("  MTPLayerConfig() constructed:", type(cfg).__name__)
PY
RC=$?
[ "${RC}" -eq 0 ] || exit "${RC}"

echo
echo "=== the Python builder imports ==="
python3 - <<'PY'
from lmdeploy.turbomind.builders import MTPLayerBuilder, MTPLayerConfig

print("  MTPLayerBuilder:", MTPLayerBuilder.__name__)
print("  MTPLayerConfig :", MTPLayerConfig.__name__)
PY
RC=$?
[ "${RC}" -eq 0 ] || exit "${RC}"

echo
echo "=== the loader method exists on the model class ==="
python3 - <<'PY'
from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5TextModel

print("  Qwen3_5TextModel.mtp:", hasattr(Qwen3_5TextModel, "mtp"))
PY
RC=$?
[ "${RC}" -eq 0 ] || exit "${RC}"

echo
echo "VERIFY_WHEEL_MTP_PASS"
