#!/usr/bin/env bash
# Report which wheels exist and whether any carries the MTP work.
#
# The build job failed after 47 minutes and its pod is gone, so the log is no
# longer readable. A wheel may still have been produced and installed before
# the failing step, and that distinction decides the next action: rerun the
# whole build, or only the verification that failed.
set -uo pipefail

echo "=== wheels present ==="
find /wheels -maxdepth 1 -name '*.whl' -printf '  %TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null |
    sort | grep . || echo "  none"

echo
echo "=== staged source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

echo
echo "=== does the newest wheel carry the MTP binding? ==="
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "${WHEEL}" ]; then
    echo "  no lmdeploy wheel to inspect"
    exit 0
fi
echo "  newest: ${WHEEL}"

# Read the wheel without installing it. The Python sources are stored verbatim
# inside, so the loader and builder can be confirmed directly.
python3 - "${WHEEL}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()

    def has(path):
        return any(n.endswith(path) for n in names)

    print("  lmdeploy/turbomind/builders/mtp_layer.py:", has("turbomind/builders/mtp_layer.py"))

    hits = [n for n in names if n.endswith("turbomind/models/qwen3_5.py")]
    if hits:
        src = z.read(hits[0]).decode("utf-8", "replace")
        print("  qwen3_5.py defines mtp():", "def mtp(self, pfx)" in src)
    else:
        print("  qwen3_5.py: ABSENT")

    so = [n for n in names if n.endswith(".so") and "turbomind" in n]
    print(f"  compiled extensions in wheel: {len(so)}")
    for n in so[:5]:
        print(f"    {n}")
PY

echo
echo "=== is a wheel currently installed, and does it expose MTPLayerConfig? ==="
cd /
python3 - <<'PY'
import importlib.util as u

spec = u.find_spec("_turbomind")
print("  _turbomind:", spec.origin if spec else "NOT INSTALLED")
if spec:
    import _turbomind as t
    print("  MTPLayerConfig:", hasattr(t, "MTPLayerConfig"))
PY
