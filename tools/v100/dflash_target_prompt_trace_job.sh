#!/usr/bin/env bash
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-target-prompt-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS/trace"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
 grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80; exit 2;
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
unset TM_DFLASH_PARITY_DIR
TM_DFLASH_TARGET_TRAJECTORY=1 TM_DFLASH_TARGET_PARITY_DIR="$RESULTS/trace" \
python3 /job/bench_decode.py \
 --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 \
 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
 --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
 --input-tokens 1000 --output-tokens 16 --trials 1 --sglang-corpus /sglang-corpus \
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
 --cache-max-entry-count 0.05 --json-out "$RESULTS/run.json" 2>&1 | tee "$RESULTS/run.log"
python3 - "$RESULTS/trace/lmdeploy" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);dirs=sorted(p for p in root.glob('rank-*-pid-*') if p.is_dir())
assert len(dirs)==4,dirs
for d in dirs:
 r={x['name']:x for x in map(json.loads,open(d/'manifest.jsonl'))}
 t=r['target.trajectory'];ids=r['target.input_ids'];f=r['target.prompt_features']
 assert t['shape']==[38,5120] and t['position']==999 and t['token_id']==198 and t['input_length']==1000,t
 assert ids['shape']==[1000] and f['shape']==[5,5120],(ids,f)
 assert (d/t['file']).stat().st_size==38*5120*2
print('DFLASH_TARGET_PROMPT_TRACE_PASS ranks=4 position=999 token=198')
PY
touch "$RESULTS/completed"
echo DFLASH_TARGET_PROMPT_TRACE_COMPLETE
