#!/usr/bin/env bash
set -euo pipefail

SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-tilelang-performance-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; [ -f "$RESULTS/completed" ] || echo KILLED >"$RESULTS/incomplete"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

cat /src/SOURCE_STAMP
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
  grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
  exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
sha256sum "$WHEEL" | tee "$RESULTS/wheel.sha256"
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

export TM_LOG_LEVEL=INFO
export TM_DFLASH_TILELANG_DRAFT_ATTENTION=1
export TM_DFLASH_DRAFT_GRAPH=1
export TM_DFLASH_GRAPH_TRACE=1
export TM_DFLASH_PERSISTENT_WORKSPACE=1
export TM_DFLASH_LOCAL_TOPK=0
export TM_DFLASH_PAGED_Q8=0
export TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1

common=(
  --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
  --tp "${TP:-4}"
  --speculative-algorithm dflash2
  --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
  --speculative-dflash-block-size 8
  --speculative-draft-window 2048
  --input-tokens 1000
  --output-tokens 256
  --trials 5
  --sglang-corpus /sglang-corpus
  --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
  --cache-max-entry-count 0.05
)

python3 /job/bench_decode.py "${common[@]}" --num-draft-tokens 0 --json-out "$RESULTS/k0.json" 2>&1 |
  tee "$RESULTS/k0.log"
python3 /job/bench_decode.py "${common[@]}" --num-draft-tokens 7 --json-out "$RESULTS/k7.json" 2>&1 |
  tee "$RESULTS/k7.log"

python3 - "$RESULTS" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
k0 = json.loads((root / 'k0.json').read_text())
k7 = json.loads((root / 'k7.json').read_text())
for name, data in [('k0', k0), ('k7', k7)]:
    assert len(data['trials']) == 5, (name, data['trials'])
    assert all(row['output_tokens'] == 256 and not row['degenerate'] for row in data['trials']), name
matches = re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',
                     (root / 'k7.log').read_text(errors='replace'))
assert matches, 'missing K=7 committed-length telemetry'
commit_lengths = {float(row[0]) for row in matches}
raw_lengths = {float(row[1]) for row in matches}
steps = {int(row[2]) for row in matches}
assert len(commit_lengths) == len(raw_lengths) == len(steps) == 1, (commit_lengths, raw_lengths, steps)
commit = commit_lengths.pop()
raw = raw_lengths.pop()
decode0 = float(k0['mean_decode_tok_s'])
decode7 = float(k7['mean_decode_tok_s'])
cycle_ms = 1000.0 * commit / decode7
ratio = decode7 / decode0
summary = {
    'k0_decode_tok_s': decode0,
    'k7_decode_tok_s': decode7,
    'matched_throughput_ratio': ratio,
    'commit_length': commit,
    'raw_commit_length': raw,
    'normalized_cycle_ms': cycle_ms,
    'required_commit_length': 3.5,
    'required_max_cycle_ms': 30.0,
    'required_min_throughput_ratio': 2.2,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print('DFLASH_TILELANG_PERFORMANCE_RESULT', json.dumps(summary, sort_keys=True))
assert commit >= 3.5, f'committed length {commit:.3f} < 3.5'
assert cycle_ms <= 30.0, f'normalized cycle {cycle_ms:.3f} ms > 30 ms'
assert ratio >= 2.2, f'matched throughput ratio {ratio:.3f}x < 2.2x'
PY

touch "$RESULTS/completed"
echo DFLASH_TILELANG_PERFORMANCE_QUALIFIED
