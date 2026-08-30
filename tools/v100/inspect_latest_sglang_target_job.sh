#!/usr/bin/env bash
# Inspect the latest read-only SGLang target trace after a failed comparison.
set -euo pipefail
ROOT="$(find /results -maxdepth 1 -type d -name '*-sglang-dflash-parity-*' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${ROOT}" ] || { echo 'FAIL: no SGLang parity result' >&2; exit 2; }
echo "SGLANG_TARGET_RESULT=${ROOT}"
echo '=== pinned qwen classes ==='
PYTHONPATH=/opt/sglang/python python3 - <<'PY' || true
import inspect
import sglang.srt.models.qwen3_5 as module
print('module_file=' + str(module.__file__))
for name in dir(module):
    value = getattr(module, name)
    if name.startswith('Qwen') and inspect.isclass(value):
        print(f'{name} module={value.__module__}')
PY
grep -En '^class |^    def (forward|__init__)' /opt/sglang/python/sglang/srt/models/qwen3_5.py | head -160 || true
echo '=== model and hook lines ==='
grep -Ein 'Qwen|model(_cls| class| architecture)|DFLASH_TARGET|Application startup|Prefill batch' "${ROOT}/server.log" | head -200 || true
echo '=== server tail ==='
tail -200 "${ROOT}/server.log"
echo '=== trajectory finite audit ==='
python3 - "${ROOT}/trace/sglang" <<'PY'
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
names = ["target.input.embedding", "target.layer0.attn_norm"]
for layer in range(6):
    names.extend(
        [
            f"target.layer{layer}.branch",
            f"target.layer{layer}.post_attn_residual",
            f"target.layer{layer}.mlp_norm",
            f"target.layer{layer}.mlp_output",
            f"target.layer{layer}.output_residual",
            f"target.layer{layer}.next_attn_norm",
        ]
    )
for directory in sorted(root.glob("rank-*-pid-*")):
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    by_name = {record["name"]: record for record in records}
    trajectory_record = by_name["target.trajectory"]
    trajectory = np.fromfile(directory / trajectory_record["file"], dtype="<f2").reshape(38, 5120)
    dtype_record = by_name.get("target.trajectory_dtypes")
    dtypes = np.fromfile(directory / dtype_record["file"], dtype="<i4") if dtype_record else np.zeros(38, dtype=np.int32)
    bad = []
    for index, name in enumerate(names):
        finite = np.isfinite(trajectory[index])
        if not finite.all():
            bad.append({"index": index, "name": name, "nonfinite": int((~finite).sum()), "dtype": int(dtypes[index])})
    target_record = by_name["target.post_layer_residual"]
    target = np.fromfile(directory / target_record["file"], dtype="<f2")
    print(json.dumps({"directory": directory.name, "bad": bad, "target_nonfinite": int((~np.isfinite(target)).sum())}))
PY
