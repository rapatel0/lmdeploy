#!/usr/bin/env bash
# Capture all TurboMind draft boundaries under SGLang's exact post-communication context.
set -euo pipefail
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-actual-context-parity
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
SG_ROOT=""
while IFS= read -r candidate; do
   if grep -q 'layer0.input.hidden_pre_norm' "$candidate"/rank-*/manifest.jsonl 2>/dev/null; then
      SG_ROOT=$candidate
      break
   fi
done < <(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' | sort -r)
[ -n "$SG_ROOT" ]
mapfile -t REPLAY < <(
   python3 - "$SG_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);ds=sorted(p for p in root.glob('rank-*-pid-*') if p.is_dir());assert len(ds)==4,ds
for name,shape in (('target.post_layer_residual',[1,25600]),('layer0.attention.norm_output',[8,5120])):
 paths=[];hashes=[]
 for d in ds:
  r={x['name']:x for x in map(json.loads,(d/'manifest.jsonl').read_text().splitlines())}[name]
  assert r['dtype']=='f16' and r['shape']==shape,r
  p=d/r['file'];paths.append(p);hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
 assert len(set(hashes))==1,(name,hashes)
 print(paths[0])
PY
)
[ "${#REPLAY[@]}" -eq 2 ]
CONTEXT=${REPLAY[0]}
INITIAL_NORM=${REPLAY[1]}
echo "sglang=$SG_ROOT context=$CONTEXT initial_norm=$INITIAL_NORM"
TM_DFLASH_CONTEXT_REPLAY_FILE="$CONTEXT" TM_DFLASH_CONTEXT_REPLAY_FRONTIER_ONLY=1 \
   TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE="$INITIAL_NORM" \
   TM_DFLASH_REDUCE_BEFORE_CONV=1 TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
   python3 /job/bench_decode.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
   --num-draft-tokens 7 --speculative-algorithm dflash2 \
   --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
   --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
   --input-tokens 1000 --output-tokens 64 --trials 1 --sglang-corpus /sglang-corpus \
   --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
   --cache-max-entry-count 0.05 --json-out "$RESULTS/bench.json" 2>&1 | tee "$RESULTS/bench.log"
[ "$(grep -c 'replaying parity target context rows=1' "$RESULTS/bench.log")" -eq 4 ]
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
