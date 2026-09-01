#!/usr/bin/env bash
# Measure production acceptance for draft reduction and context-rounding policies.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-policy-matrix-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

common=(
  --model /models/Qwen3.8-27B-FP8 --tp 4 --num-draft-tokens 7
  --speculative-algorithm dflash2 --speculative-draft-model /models/Qwen3.8-27B-DFlash2
  --speculative-dflash-block-size 8 --speculative-draft-window 2048
  --input-tokens 1000 --output-tokens 128 --trials 1 --sglang-corpus /sglang-corpus
  --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
  --cache-max-entry-count 0.05
)

for bits in 000 100 010 001 110 101 011 111; do
  reduce=${bits:0:1}
  ordered=${bits:1:1}
  bf16=${bits:2:1}
  name="r${reduce}-o${ordered}-b${bf16}"
  echo "DFLASH_POLICY_MATRIX_BEGIN $name"
  env \
    TM_DFLASH_TILELANG_DRAFT_ATTENTION=1 \
    TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
    TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
    TM_DFLASH_REDUCE_BEFORE_CONV="$reduce" \
    TM_DFLASH_RANK_ORDERED_ALLREDUCE="$ordered" \
    TM_DFLASH_CONTEXT_BF16_ROUND="$bf16" \
    python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/$name.json" \
    2>&1 | tee "$RESULTS/$name.log"
done

python3 - "$RESULTS" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for path in sorted(root.glob('r*-o*-b*.log')):
    text=path.read_text(errors='replace')
    matches=re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',text)
    assert matches,path
    unique=set(matches); assert len(unique)==1,(path,unique)
    commit,raw,steps=unique.pop(); bench=json.loads(path.with_suffix('.json').read_text())
    rows.append({'name':path.stem,'commit_length':float(commit),'raw_commit_length':float(raw),
                 'verification_steps':int(steps),'decode_tok_s':float(bench['mean_decode_tok_s'])})
rows.sort(key=lambda x:(-x['commit_length'],-x['decode_tok_s']))
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows: print('DFLASH_POLICY_MATRIX_RESULT',json.dumps(row,sort_keys=True))
print('DFLASH_POLICY_MATRIX_BEST',json.dumps(rows[0],sort_keys=True))
PY

touch "$RESULTS/completed"
echo DFLASH_POLICY_MATRIX_COMPLETE
