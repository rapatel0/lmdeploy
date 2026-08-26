#!/usr/bin/env bash
# Report what the island already has, before any build or benchmark.
#
# Three questions decide the next step:
#   1. Which commit is the staged source at, and does it carry the MTP work?
#   2. Is there a built wheel, and is it older than those commits?
#   3. Is the FP8 model present, and does it carry the mtp. tensors?
set -uo pipefail

echo "=== 1. staged source ==="
cd /src 2>/dev/null || {
    echo "FATAL: /src is not mounted"
    exit 2
}
git log --oneline -3 2>/dev/null || echo "  (not a git tree)"
echo
echo "--- MTP files present in the staged source? ---"
for f in src/turbomind/models/mtp_weight.h \
    src/turbomind/models/mtp_weight.cc \
    lmdeploy/turbomind/builders/mtp_layer.py; do
    if [ -f "$f" ]; then echo "  PRESENT $f"; else echo "  ABSENT  $f"; fi
done
echo
echo "--- is the mtp child wired into ModelWeight? ---"
grep -c "MTPLayerWeight, mtp" src/turbomind/models/model_weight.h 2>/dev/null |
    sed 's/^/  X(MTPLayerWeight, mtp) occurrences: /'

echo
echo "=== 2. wheels ==="
find /wheels -maxdepth 1 -name '*.whl' -printf '  %TY-%Tm-%Td %10s %p\n' 2>/dev/null |
    sort | tail -5 | grep . || echo "  no wheels"

echo
echo "=== 3. installed runtime ==="
python3 -c "
import importlib.util as u
s = u.find_spec('lmdeploy')
print('  lmdeploy:', s.origin if s else 'NOT INSTALLED')
s2 = u.find_spec('_turbomind')
print('  _turbomind:', s2.origin if s2 else 'NOT INSTALLED')
" 2>/dev/null

echo
echo "--- does the INSTALLED _turbomind expose MTPLayerConfig? ---"
python3 -c "
try:
    import _turbomind as t
    print('  MTPLayerConfig:', hasattr(t, 'MTPLayerConfig'))
    print('  DecoderLayerConfig:', hasattr(t, 'DecoderLayerConfig'))
except Exception as e:
    print('  import failed:', type(e).__name__, e)
" 2>/dev/null

echo
echo "=== 4. model store ==="
find /models -maxdepth 1 -mindepth 1 -printf '  %f\n' 2>/dev/null | sort | head -20

echo
echo "--- Qwen3.8-27B FP8 candidates ---"
find /models -maxdepth 2 -iname "*qwen*" -o -maxdepth 2 -iname "*27b*" 2>/dev/null | head -10
