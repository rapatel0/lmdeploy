#!/usr/bin/env bash
set -euo pipefail

SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-fp16-shadow-${SRC_COMMIT:-unknown}
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
    --trials 3
    --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05
)

for arm in baseline shadow; do
    env_args=()
    if [ "$arm" = shadow ]; then
        env_args+=(TM_SM70_FP8_FP16_SHADOW=1)
    fi
    env "${env_args[@]}" python3 /job/bench_decode.py "${common[@]}" --num-draft-tokens 0 \
        --json-out "$RESULTS/${arm}_k0.json" 2>&1 | tee "$RESULTS/${arm}_k0.log"
    env "${env_args[@]}" python3 /job/bench_decode.py "${common[@]}" --num-draft-tokens 7 \
        --json-out "$RESULTS/${arm}_k7.json" 2>&1 | tee "$RESULTS/${arm}_k7.log"
done

[ "$(grep -c 'SM70_FP8_FP16_SHADOW_REGISTERED device=' "$RESULTS/shadow_k7.log" || true)" -eq 4 ]
[ "$(grep -c 'SM70_FP8_FP16_SHADOW_ACTIVE device=' "$RESULTS/shadow_k7.log" || true)" -eq 4 ]

TM_SM70_FP8_FP16_SHADOW=1 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 |
    tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"

python3 - "$RESULTS" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for arm in ('baseline', 'shadow'):
    k0 = json.loads((root / f'{arm}_k0.json').read_text())
    k7 = json.loads((root / f'{arm}_k7.json').read_text())
    assert len(k0['trials']) == len(k7['trials']) == 3
    assert all(not row['degenerate'] and row['output_tokens'] == 256 for row in k0['trials'] + k7['trials'])
    matches = re.findall(
        r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',
        (root / f'{arm}_k7.log').read_text(errors='replace'))
    assert matches and all(value == matches[0] for value in matches), (arm, matches)
    commit, raw, steps = matches[0]
    decode0 = float(k0['mean_decode_tok_s'])
    decode7 = float(k7['mean_decode_tok_s'])
    rows.append({
        'arm': arm,
        'commit_length': float(commit),
        'raw_commit_length': float(raw),
        'steps': int(steps),
        'k0_decode_tok_s': decode0,
        'k7_decode_tok_s': decode7,
        'ratio': decode7 / decode0,
        'normalized_cycle_ms': 1000.0 * float(commit) / decode7,
    })
base, shadow = rows
shadow['cycle_change_pct'] = 100.0 * (shadow['normalized_cycle_ms'] / base['normalized_cycle_ms'] - 1.0)
(root / 'summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
for row in rows:
    print('DFLASH_FP8_FP16_SHADOW_RESULT', json.dumps(row, sort_keys=True))
assert shadow['normalized_cycle_ms'] <= base['normalized_cycle_ms'] * 0.99, (
    'shadow failed the 1% cycle-retention gate', rows)
PY

touch "$RESULTS/completed"
echo DFLASH_FP8_FP16_SHADOW_COMPLETE
