#!/usr/bin/env bash
# Compare the native prompt-frontier TurboMind block with native SGLang TP4.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-native-first-block-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS/parity"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; [ -f "$RESULTS/completed" ] || echo KILLED >"$RESULTS/incomplete"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
        grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
        exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

SG_ROOT=$(
        python3 - <<'PY'
import json,pathlib
matches=[]
for root in pathlib.Path('/results').glob('*-sglang-dflash-parity-*/trace/sglang'):
    dirs=sorted(p for p in root.glob('rank-*-pid-*') if p.is_dir())
    if len(dirs)!=4: continue
    try:
        for directory in dirs:
            rows=list(map(json.loads,(directory/'manifest.jsonl').read_text().splitlines()))
            names=[row['name'] for row in rows]
            if len(names)!=len(set(names)): raise ValueError('duplicates')
            records={row['name']:row for row in rows}
            block=records['block.ids']
            top=records['target.next_token_top16_ids']
            if block.get('draft_block_index')!=block.get('capture_block_index'): raise ValueError('uncorrelated')
            if top['shape']!=[16]: raise ValueError('top16')
        matches.append(root)
    except (KeyError,OSError,ValueError,json.JSONDecodeError):
        continue
assert matches,'no native correlated SGLang trace exists'
print(sorted(matches)[-1])
PY
)
echo "sglang=$SG_ROOT"

TM_DFLASH_DRAFT_AFTER_PREFILL=1 \
        TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
        TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
        TM_DFLASH_TILELANG_DRAFT_ATTENTION=1 \
        TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
        python3 /job/bench_decode.py \
        --model /models/Qwen3.8-27B-FP8 --tp 4 --num-draft-tokens 7 \
        --speculative-algorithm dflash2 --speculative-draft-model /models/Qwen3.8-27B-DFlash2 \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 64 --trials 1 --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "$RESULTS/bench.json" 2>&1 | tee "$RESULTS/bench.log"

grep -q 'DFLASH_TILELANG_SELECTOR selected=true' "$RESULTS/bench.log"
grep -q 'DFLASH_METADATA_REBUILD_ACTIVE' "$RESULTS/bench.log"
python3 /job/compare_dflash_attention_layers.py \
        --lmdeploy "$RESULTS/parity/lmdeploy" --sglang "$SG_ROOT" \
        --output "$RESULTS/attention.json" | tee "$RESULTS/attention.log"
python3 /job/compare_dflash_parity.py \
        --lmdeploy "$RESULTS/parity/lmdeploy" --sglang "$SG_ROOT" \
        --output "$RESULTS/model.json" | tee "$RESULTS/model.log"

touch "$RESULTS/completed"
echo DFLASH_NATIVE_FIRST_BLOCK_PARITY_COMPLETE
