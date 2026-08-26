#!/usr/bin/env bash
# Load Qwen3.8-27B-FP8 and prove the MTP layer is consumed.
#
# Everything so far proves the code is present. This proves it runs: that the
# 15 mtp. tensors are read, that the new log line fires, and that adding a
# child to ModelWeight did not disturb the existing weight tree.
#
# The last point is the reason this job generates text. ModelWeight gained a
# member and an out-of-line destructor, and a mistake there corrupts unrelated
# weights rather than MTP ones, surfacing as garbled output instead of a load
# error.
set -uo pipefail

MODEL=/models/Qwen3.8-27B-FP8

echo "=== source stamp ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "  no stamp"

echo
echo "=== install the built wheel ==="
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

echo
echo "=== run the load verification ==="
# Run from / so /src/lmdeploy cannot shadow the installed package.
cd /
# TM_LOG_LEVEL surfaces the C++ TM_LOG_INFO line from MTPLayerWeight::verify.
export TM_LOG_LEVEL=INFO
python3 /job/verify_mtp_load.py \
    --model "${MODEL}" \
    --tp 4 \
    --model-format fp8 2>&1
RC=$?

echo
echo "=== exit ${RC} ==="
exit "${RC}"
