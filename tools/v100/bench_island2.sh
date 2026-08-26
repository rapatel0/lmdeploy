#!/usr/bin/env bash
# Measure single-request TP4 FP8 decode against SGLang's audited Qwen3.8 number.
#
# Reference: sglang-V100/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822
#   1,024 input, 256 output, TP4, one request at a time: 58.21 tok/s.
#
# Runs on island 2 only. run_island2.sh enforces that.
set -euo pipefail

MODEL=/models/Qwen3.8-27B-FP8
# /job is a read-only ConfigMap mount, so a result written there is discarded
# with only a warning. Write to /wheels, which is a writable hostPath and
# therefore outlives the pod.
OUT=/wheels/bench-tp4-fp8.json

echo "=== visible GPUs ==="
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader

echo
echo "=== install the built wheel ==="
WHEEL=$(ls -t /wheels/lmdeploy-*.whl | head -1)
echo "  using $WHEEL"
pip install --no-deps --force-reinstall "$WHEEL" >/tmp/pip.log 2>&1 || {
    tail -20 /tmp/pip.log
    exit 3
}

echo
echo "=== wheel in use ==="
python3 -c "import lmdeploy, os; print(lmdeploy.__version__, os.path.dirname(lmdeploy.__file__))"

echo
echo "=== confirm the FP8 scale cast is in the packaged source ==="
python3 - <<'PY'
import inspect

from lmdeploy.turbomind.weight_format import FP8Format

src = inspect.getsource(FP8Format)
print("  scale_dtype present   :", "scale_dtype" in src)
print("  check_scale_range     :", "check_scale_range" in src)
PY

echo
echo "=== benchmark: TP4, FP8, 1024 in, 256 out ==="
python3 /job/bench_decode.py \
    --model "$MODEL" \
    --model-format fp8 \
    --tp 4 \
    --input-tokens 1024 \
    --output-tokens 256 \
    --trials 3 \
    --require-mtp \
    --json-out "$OUT"

echo
echo "=== result file ==="
cat "$OUT"
