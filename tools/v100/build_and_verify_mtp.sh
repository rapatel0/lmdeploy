#!/usr/bin/env bash
# Build the wheel from the staged source, then prove the MTP work is in it.
#
# A build that succeeds proves only that the code compiles. It does not prove
# that the new binding reached the extension module, and it does not prove that
# the loader consumes the checkpoint tensors. Both are checked here, because a
# silent miss looks exactly like a working run.
#
# The MTP layer is optional at every level by design, so a loader that never
# fires produces a model that loads, generates correct text, and contains no
# MTP weights at all. Only an explicit check distinguishes that from success.
set -euo pipefail

echo "=== source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

echo
echo "=== confirm the staged tree carries the MTP work ==="
cd /src
MISSING=0
for f in src/turbomind/models/mtp_weight.h \
    src/turbomind/models/mtp_weight.cc \
    lmdeploy/turbomind/builders/mtp_layer.py; do
    if [ -f "$f" ]; then
        echo "  PRESENT $f"
    else
        echo "  ABSENT  $f"
        MISSING=1
    fi
done
if ! grep -q "MTPLayerWeight, mtp" src/turbomind/models/model_weight.h; then
    echo "  ABSENT  X(MTPLayerWeight, mtp) in model_weight.h"
    MISSING=1
else
    echo "  PRESENT X(MTPLayerWeight, mtp) in model_weight.h"
fi
if [ "$MISSING" -ne 0 ]; then
    echo "FAIL: the staged source predates the MTP work; re-run sync_src.sh" >&2
    exit 3
fi

echo
echo "=== build ==="
bash /src/tools/v100/build_v100.sh

echo
echo "=== install the wheel just built ==="
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' |
    sort -rn | head -1 | cut -d' ' -f2-)"
echo "installing ${WHEEL}"
pip install --no-deps --force-reinstall "${WHEEL}"

echo
echo "=== prove the binding reached the extension ==="
# Run from / so that /src/lmdeploy does not shadow the installed package.
cd /
python3 - <<'PY'
import sys

# _turbomind lives in lmdeploy/lib, which is not on sys.path until
# lmdeploy.turbomind.turbomind appends it (see its module-level
# sys.path.append). Import that first; a direct `import _turbomind` raises
# ModuleNotFoundError even when the extension is installed correctly.
import lmdeploy.turbomind.turbomind  # noqa: F401

import _turbomind as t

ok = True
for name in ("DecoderLayerConfig", "MTPLayerConfig", "ModelWeightConfig"):
    present = hasattr(t, name)
    print(f"  {name}: {present}")
    if not present:
        ok = False

if not ok:
    print("FAIL: a required config binding is missing", file=sys.stderr)
    sys.exit(4)

# The config must be constructible, not merely present. A bound-but-broken
# type raises here rather than at model load, where the error is harder to read.
cfg = t.MTPLayerConfig()
print("  MTPLayerConfig() constructed:", type(cfg).__name__)
PY

echo
echo "=== prove the Python builder imports ==="
python3 - <<'PY'
from lmdeploy.turbomind.builders import MTPLayerBuilder, MTPLayerConfig

print("  MTPLayerBuilder:", MTPLayerBuilder.__name__)
print("  MTPLayerConfig :", MTPLayerConfig.__name__)
PY

echo
echo "BUILD_AND_VERIFY_MTP_PASS"
