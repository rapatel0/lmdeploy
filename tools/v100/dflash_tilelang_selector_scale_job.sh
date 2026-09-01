#!/usr/bin/env bash
set -euo pipefail

SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-tilelang-selector-scale-${SRC_COMMIT:-unknown}
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
export TM_DFLASH_PERSISTENT_WORKSPACE=1
export TM_DFLASH_LOCAL_TOPK=0
export TM_DFLASH_PAGED_Q8=0
export TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1

common=(
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
    --tp "${TP:-4}"
    --num-draft-tokens 7
    --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8
    --speculative-draft-window 2048
    --input-tokens 1000
    --output-tokens 256
    --trials 1
    --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05
)

for scale in 0 0.5 1 2 4 8; do
    tag=${scale/./p}
    TM_DFLASH_SELECTOR_TRANSITION_SCALE="$scale" \
        python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/scale_${tag}.json" 2>&1 |
        tee "$RESULTS/scale_${tag}.log"
done

python3 - "$RESULTS" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('scale_*.log')):
    tag = path.stem.removeprefix('scale_')
    scale = float(tag.replace('p', '.'))
    text = path.read_text(errors='replace')
    matches = re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps', text)
    assert matches, path
    commit, raw, steps = matches[0]
    assert all(row == matches[0] for row in matches), (path, matches)
    data = json.loads((root / f'scale_{tag}.json').read_text())
    decode = float(data['mean_decode_tok_s'])
    rows.append({
        'scale': scale,
        'commit_length': float(commit),
        'raw_commit_length': float(raw),
        'steps': int(steps),
        'decode_tok_s': decode,
        'normalized_cycle_ms': 1000.0 * float(commit) / decode,
    })
rows.sort(key=lambda row: row['scale'])
(root / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
for row in rows:
    print('DFLASH_TILELANG_SELECTOR_SCALE_RESULT', json.dumps(row, sort_keys=True))
PY

touch "$RESULTS/completed"
echo DFLASH_TILELANG_SELECTOR_SCALE_COMPLETE
