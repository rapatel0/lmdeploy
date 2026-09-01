#!/usr/bin/env bash
# Capture all TurboMind draft boundaries under SGLang's exact post-communication context.
set -euo pipefail
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-actual-context-parity
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
SOURCE_COMMIT=$(sed -n 's/^commit=\(.*\)/\1/p' /src/SOURCE_STAMP)
WHEEL_COMMIT=$(sed -n 's/^commit=\(.*\)/\1/p' /wheels/SOURCE_STAMP 2>/dev/null || true)
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$WHEEL" ] || [ "$SOURCE_COMMIT" != "$WHEEL_COMMIT" ]; then
 rm -f /wheels/lmdeploy-*.whl
 bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
  grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
  exit 2
 }
 WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
else
 echo "reusing provenance-matched wheel commit=$WHEEL_COMMIT"
fi
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
SG_ROOT=$(
 python3 - <<'PY'
import array,json,pathlib
matches=[]
for root in pathlib.Path('/results').glob('*-sglang-dflash-parity-*/trace/sglang'):
 ds=sorted(p for p in root.glob('rank-*-pid-*') if p.is_dir())
 if len(ds)!=4:continue
 try:
  records=[]
  for d in ds:
   rows=list(map(json.loads,(d/'manifest.jsonl').read_text().splitlines()))
   names=[x['name'] for x in rows]
   if len(names)!=len(set(names)):raise ValueError('duplicate records')
   records.append({x['name']:x for x in rows})
  if not all('layer0.input.hidden_pre_norm' in r and 'target.next_token_top16_ids' in r for r in records):continue
  ids=[]
  for d,r in zip(ds,records):
   x=r['block.ids'];code='q' if x['dtype']=='i64' else 'i';v=array.array(code);v.frombytes((d/x['file']).read_bytes());block=list(v)
   top=r['target.next_token_top16_ids'];t=array.array('q');t.frombytes((d/top['file']).read_bytes())
   if block != [int(t[0])]+[248070]*7:raise ValueError((block,int(t[0])))
   if x.get('draft_block_index')!=x.get('capture_block_index'):raise ValueError(x)
   ids.append(block)
  if all(v==ids[0] for v in ids):matches.append(root)
 except Exception:continue
assert matches,'no complete native SGLang trace has a target-correlated block'
print(sorted(matches)[-1])
PY
)
[ -n "$SG_ROOT" ]
mapfile -t REPLAY < <(
 python3 - "$SG_ROOT" "$RESULTS" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);ds=sorted(p for p in root.glob('rank-*-pid-*') if p.is_dir());assert len(ds)==4,ds
records=[{x['name']:x for x in map(json.loads,(d/'manifest.jsonl').read_text().splitlines())} for d in ds]
frontier=(ds[0]/records[0]['target.post_layer_residual']['file']).read_bytes()
assert all((d/r['target.post_layer_residual']['file']).read_bytes()==frontier for d,r in zip(ds,records))
initial=[]
for d,r in zip(ds,records):
 x=r['layer0.attention.norm_output'];assert x['dtype']=='f16' and x['shape']==[8,5120],x;initial.append(d/x['file'])
assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in initial})==1
matches=[]
for manifest in pathlib.Path('/results').glob('*-sglang-dflash-parity-*/trace/sglang/rank-*/manifest.jsonl'):
 try:r={x['name']:x for x in map(json.loads,manifest.read_text().splitlines())}
 except Exception:continue
 if 'target.full_context' not in r:continue
 x=r['target.full_context'];p=manifest.parent/x['file']
 if x.get('dtype')=='f16' and x.get('shape')==[1000,25600] and p.read_bytes()[-len(frontier):]==frontier:
  matches.append(manifest.parent.parent)
assert matches,'no SGLang full context has the exact detailed frontier row'
full_root=sorted(set(matches))[-1];full_ds=sorted(p for p in full_root.glob('rank-*-pid-*') if p.is_dir());assert len(full_ds)==4
full=[]
for d in full_ds:
 r={x['name']:x for x in map(json.loads,(d/'manifest.jsonl').read_text().splitlines())}['target.full_context'];full.append(d/r['file'])
assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in full})==1
print(full[0]);print(initial[0])
PY
)
[ "${#REPLAY[@]}" -eq 2 ]
CONTEXT=${REPLAY[0]}
INITIAL_NORM=${REPLAY[1]}
echo "sglang=$SG_ROOT full_context=$CONTEXT initial_norm=$INITIAL_NORM"
TM_DFLASH_CONTEXT_REPLAY_FILE="$CONTEXT" \
 TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE="$INITIAL_NORM" \
 TM_DFLASH_REDUCE_BEFORE_CONV=1 TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
 TM_DFLASH_ASSERT_DRAFT_METADATA=1 TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
 python3 /job/bench_decode.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
 --num-draft-tokens 7 --speculative-algorithm dflash2 \
 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
 --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
 --input-tokens 1000 --output-tokens 64 --trials 1 --sglang-corpus /sglang-corpus \
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
 --cache-max-entry-count 0.05 --json-out "$RESULTS/bench.json" 2>&1 | tee "$RESULTS/bench.log"
[ "$(grep -c 'replaying parity target context rows=1000' "$RESULTS/bench.log")" -eq 4 ]
[ "$(grep -c 'replaying parity target context rows=1 from' "$RESULTS/bench.log")" -eq 4 ]
[ "$(grep -c 'TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE' "$RESULTS/bench.log")" -eq 4 ]
python3 /job/compare_dflash_parity.py --lmdeploy "$RESULTS/parity/lmdeploy" --sglang "$SG_ROOT" \
 --output "$RESULTS/compare.json" | tee "$RESULTS/compare.log"
python3 - "$RESULTS/compare.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print('EARLIEST',r['earliest_mismatch'])
for x in r['comparisons']:
 print('BOUNDARY',x['lmdeploy'],x['status'],'max',x.get('max_abs'),'rms',x.get('rms'))
PY
touch "$RESULTS/completed"
echo DFLASH_ACTUAL_CONTEXT_PARITY_COMPLETE
