#!/usr/bin/env bash
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-tilelang-nccl-matrix-${SRC_COMMIT:-unknown}
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
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO TM_DFLASH_TILELANG_DRAFT_ATTENTION=1 TM_DFLASH_DRAFT_GRAPH=1
export TM_DFLASH_PERSISTENT_WORKSPACE=1 TM_DFLASH_LOCAL_TOPK=0 TM_DFLASH_PAGED_Q8=0
export TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
   --input-tokens 1000 --output-tokens 256 --trials 1 --sglang-corpus /sglang-corpus
   --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
   --cache-max-entry-count 0.05)
for arm in default win0 ring_ll ring_ll128; do
   vars=()
   case "$arm" in
   default) vars=() ;;
   win0) vars=(NCCL_WIN_ENABLE=0) ;;
   ring_ll) vars=(NCCL_WIN_ENABLE=0 NCCL_ALGO=Ring NCCL_PROTO=LL) ;;
   ring_ll128) vars=(NCCL_WIN_ENABLE=0 NCCL_ALGO=Ring NCCL_PROTO=LL128) ;;
   esac
   for k in 0 7; do
      args=("${common[@]}" --num-draft-tokens "$k")
      if [ "$k" -gt 0 ]; then
         args+=(--speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
            --speculative-dflash-block-size 8 --speculative-draft-window 2048)
      fi
      env "${vars[@]}" python3 /job/bench_decode.py "${args[@]}" --json-out "$RESULTS/${arm}_k${k}.json" 2>&1 |
         tee "$RESULTS/${arm}_k${k}.log"
   done
done
python3 - "$RESULTS" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for arm in ('default','win0','ring_ll','ring_ll128'):
 k0=float(json.loads((root/f'{arm}_k0.json').read_text())['mean_decode_tok_s'])
 k7=float(json.loads((root/f'{arm}_k7.json').read_text())['mean_decode_tok_s'])
 text=(root/f'{arm}_k7.log').read_text(errors='replace')
 m=re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',text)
 assert m and all(x==m[0] for x in m),(arm,m)
 commit,raw,steps=m[0]
 row={'arm':arm,'k0_decode_tok_s':k0,'k7_decode_tok_s':k7,'ratio':k7/k0,'commit_length':float(commit),'raw_commit_length':float(raw),'steps':int(steps),'normalized_cycle_ms':1000*float(commit)/k7}
 rows.append(row); print('DFLASH_TILELANG_NCCL_RESULT',json.dumps(row,sort_keys=True))
(root/'summary.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
PY
touch "$RESULTS/completed"
echo DFLASH_TILELANG_NCCL_MATRIX_COMPLETE
