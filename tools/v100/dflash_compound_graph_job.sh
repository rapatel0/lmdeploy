#!/usr/bin/env bash
# Retest the retained draft graph after all qualified default optimizations stack.
set -euo pipefail
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-compound-graph
mkdir -p "${RESULTS}"; exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n'|sort -rn|head -1|cut -d' ' -f2-)"; [ -n "${WHEEL}" ]
sha256sum "${WHEEL}"|tee "${RESULTS}/wheel.sha256"; pip install --no-deps --force-reinstall "${WHEEL}" 2>&1|tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256 --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 --cache-max-entry-count 0.05)
for spec in control_a:0 graph_a:1 graph_b:1 control_b:0; do
 name=${spec%%:*}; graph=${spec##*:}
 TM_DFLASH_DRAFT_GRAPH="${graph}" TM_DFLASH_GRAPH_TRACE="${graph}" python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/${name}.json" 2>&1|tee "${RESULTS}/${name}.log"
done
NSYS="$(command -v nsys 2>/dev/null || true)"; [ -n "${NSYS}" ] || NSYS=/opt/nsys/nsys; export FT_NVTX=ON
for graph in 0 1; do
 TM_DFLASH_DRAFT_GRAPH="${graph}" TM_DFLASH_GRAPH_TRACE="${graph}" "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${graph}" python3 /job/bench_decode.py "${common[@]/--output-tokens 256/--output-tokens 128}" --trials 1 --cuda-profiler-range --json-out "${RESULTS}/profile_${graph}.json" 2>&1|tee "${RESULTS}/profile_${graph}.log"
done
unset FT_NVTX
python3 - "${RESULTS}" <<'PY'
import json,re,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]); pat=re.compile(r'final commit length ([0-9.]+), raw [0-9.]+ over (\d+) verification steps')
def metric(name):
 d=json.loads((r/f'{name}.json').read_text()); assert len(d['trials']) in (1,5) and all(not x.get('degenerate') for x in d['trials']); m=pat.findall((r/f'{name}.log').read_text(errors='replace')); assert m; c=statistics.median(float(x[0]) for x in m); return {'decode_tok_s':float(d['mean_decode_tok_s']),'commit_length':c,'cycle_ms':1000*c/float(d['mean_decode_tok_s'])}
rows={n:metric(n) for n in ('control_a','graph_a','graph_b','control_b','profile_0','profile_1')}; ca=statistics.mean(rows[n]['cycle_ms'] for n in ('control_a','control_b')); ga=statistics.mean(rows[n]['cycle_ms'] for n in ('graph_a','graph_b')); rows['pooled']={'control_cycle_ms':ca,'graph_cycle_ms':ga,'graph_pct':100*(ga/ca-1),'profile_pct':100*(rows['profile_1']['cycle_ms']/rows['profile_0']['cycle_ms']-1)}
captures=(r/'graph_a.log').read_text(errors='replace').count('draft graph captured phase=')
if captures < 4: raise SystemExit(f'missing graph captures: {captures}')
(r/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n'); print('DFLASH_COMPOUND_GRAPH_ANALYSIS',json.dumps(rows,sort_keys=True))
PY
TM_DFLASH_DRAFT_GRAPH=1 python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1|tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS' "${RESULTS}/identity.log"; touch "${RESULTS}/completed"; echo DFLASH_COMPOUND_GRAPH_COMPLETE
