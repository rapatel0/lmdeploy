#!/usr/bin/env bash
# Capture TurboMind's second frozen-KV DFlash proposal and verify accepted-cache append mapping.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-second-block-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
   grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
   exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
SG=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang
mapfile -t REPLAY < <(
   python3 - "$SG" <<'PY'
import glob,json,pathlib,sys
root=sorted(glob.glob(sys.argv[1]+'/rank-*-pid-*'))[0]
rows={r['name']:r for r in map(json.loads,open(root+'/manifest.jsonl'))}
for name in ('target.full_context','context.full_norm'):
 print(pathlib.Path(root,rows[name]['file']))
PY
)
mkdir -p "$RESULTS/parity"
TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
   TM_DFLASH_PARITY_BLOCK_INDEX=2 \
   TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
   TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
   TM_DFLASH_CONTEXT_REPLAY_FULL_ONLY=1 \
   TM_DFLASH_CONTEXT_REPLAY_FILE="${REPLAY[0]}" \
   TM_DFLASH_CONTEXT_NORM_REPLAY_FILE="${REPLAY[1]}" \
   python3 /job/bench_decode.py \
   --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 \
   --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
   --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
   --input-tokens 1000 --output-tokens 16 --trials 1 --sglang-corpus /sglang-corpus \
   --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
   --cache-max-entry-count 0.05 --json-out "$RESULTS/run.json" 2>&1 | tee "$RESULTS/run.log"
python3 - "$RESULTS" <<'PY' | tee "$RESULTS/analysis.log"
import glob,json,pathlib,re,sys,numpy as np
root=pathlib.Path(sys.argv[1])
match=re.search(r'DFLASH_METADATA_REBUILD_ACTIVE device=0 .*?new_q_sum=8 new_k_sum=(\d+)',(root/'run.log').read_text(errors='replace'))
if not match: raise SystemExit('FAIL: no corrected metadata record')
# The first record is block one. Select the second record for the captured block.
spans=[int(x) for x in re.findall(r'DFLASH_METADATA_REBUILD_ACTIVE device=0 .*?new_q_sum=8 new_k_sum=(\d+)',(root/'run.log').read_text(errors='replace'))]
if len(spans)<2: raise SystemExit(f'FAIL: expected two metadata spans, got {spans}')
klen=spans[1]
print('SECOND_BLOCK_PREFIX',klen)
dirs=sorted(glob.glob(str(root/'parity/lmdeploy/rank-*-pid-*')))
if len(dirs)!=4: raise SystemExit(f'FAIL: expected four traces, got {dirs}')
for directory in dirs:
 records={r['name']:r for r in map(json.loads,open(directory+'/manifest.jsonl'))}
 def load(name):
  row=records[name];dt={'f16':'<f2','f32':'<f4','i32':'<i4'}[row['dtype']]
  return np.fromfile(directory+'/'+row['file'],dtype=dt).reshape(row['shape']).astype('f4')
 flat=load('layer0.attention.flattened_kv').reshape(-1)[:2*2*klen*128].reshape(2,2,klen,128)
 frontier=load('context.frontier.layer0.qkv_projection').reshape(8,1536)[:,1280:].reshape(8,2,128).transpose(1,0,2)
 appended=klen-1000
 if appended<=0: raise SystemExit(f'FAIL: no accepted cache rows in {directory}: klen={klen}')
 tail=flat[:,1,1000:klen]
 errors=np.sqrt(np.mean((tail[:,:,None,:]-frontier[:,None,:,:])**2,axis=3))
 best=errors.argmin(axis=2)
 best_error=np.take_along_axis(errors,best[:,:,None],axis=2)[:,:,0]
 print('SECOND_BLOCK_V_MAP',pathlib.Path(directory).name,'appended',appended,'rows',best.tolist(),'rms',best_error.tolist())
 if not np.array_equal(best,np.broadcast_to(np.arange(appended),(2,appended))):
  raise SystemExit(f'FAIL: accepted V rows do not map to frontier prefix in {directory}')
 if float(best_error.max())>0.05: raise SystemExit(f'FAIL: accepted V row RMS too high in {directory}: {best_error.max()}')
print('DFLASH_SECOND_BLOCK_CACHE_MAPPING_PASS')
PY
touch "$RESULTS/completed"
echo DFLASH_SECOND_BLOCK_TRACE_COMPLETE
