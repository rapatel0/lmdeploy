#!/usr/bin/env bash
# Qualify exact TP-local top-2 DFlash verification on V100 TP4.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-tp-local-top2-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"; exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
rm -f /wheels/lmdeploy-*.whl; started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || { grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100; exit 2; }
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n'|sort -rn|head -1|cut -d' ' -f2-)"; [ "$(stat -c %Y "${WHEEL}")" -ge "${started}" ]
sha256sum "${WHEEL}"|tee "${RESULTS}/wheel.sha256"; pip install --no-deps --force-reinstall "${WHEEL}" 2>&1|tail -1
NSYS="$(command -v nsys 2>/dev/null||true)"; [ -n "${NSYS}" ]||NSYS=/opt/nsys/nsys
python3 /job/verify_dflash_tp_local_top2.py 2>&1|tee "${RESULTS}/micro.log"; grep -q 'DFLASH_TP_LOCAL_TOP2_MICRO_PASS' "${RESULTS}/micro.log"
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256 --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 --cache-max-entry-count 0.05)
for arm in baseline top2;do if [ "$arm" = top2 ];then mode=1;else mode=0;fi; TM_DFLASH_TP_LOCAL_VERIFY_TOP2="$mode" python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/${arm}.json" 2>&1|tee "${RESULTS}/${arm}.log";done
export FT_NVTX=ON
for arm in baseline top2;do if [ "$arm" = top2 ];then mode=1;else mode=0;fi; TM_DFLASH_TP_LOCAL_VERIFY_TOP2="$mode" "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${arm}" python3 /job/bench_decode.py "${common[@]}" --output-tokens 128 --trials 1 --cuda-profiler-range --json-out "${RESULTS}/profile_${arm}.json" 2>&1|tee "${RESULTS}/profile_${arm}.log"; "${NSYS}" export --type sqlite --force-overwrite=true --output="${RESULTS}/profile_${arm}.sqlite" "${RESULTS}/profile_${arm}.nsys-rep" >/dev/null 2>&1;done
unset FT_NVTX
python3 - "${RESULTS}" <<'PY'
import json,re,sqlite3,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]);pat=re.compile(r'final commit length ([0-9.]+), raw [0-9.]+ over (\d+) verification steps')
def cyc(n):
 d=json.loads((r/f'{n}.json').read_text());rows=d.get('trials',[]);want=1 if n.startswith('profile_') else 5
 if len(rows)!=want or any(x.get('degenerate') for x in rows):raise SystemExit(f'invalid benchmark {n}')
 c=statistics.median(float(x[0]) for x in pat.findall((r/f'{n}.log').read_text(errors='replace')));return 1000*c/float(d['mean_decode_tok_s'])
def ranges(n,label):
 q=sqlite3.connect(r/f'profile_{n}.sqlite');v=q.execute("SELECT COUNT(*) FROM NVTX_EVENTS n JOIN StringIds s ON s.id=n.textId WHERE s.value=?",(label,)).fetchone()[0];q.close();return v
u0,u1=cyc('baseline'),cyc('top2');p0,p1=cyc('profile_baseline'),cyc('profile_top2');up=100*(u1/u0-1);pp=100*(p1/p0-1)
bfull=ranges('baseline','postDecodeEmbedding');tfull=ranges('top2','postDecodeEmbedding');tcompact=ranges('top2','postDecodeEmbeddingLocalTop2')
out={'baseline_cycle_ms':u0,'top2_cycle_ms':u1,'unprofiled_pct':up,'profiled_pct':pp,'baseline_full_ranges':bfull,'top2_full_ranges':tfull,'top2_compact_ranges':tcompact};(r/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('DFLASH_TP_LOCAL_TOP2_ANALYSIS',json.dumps(out,sort_keys=True))
if bfull<=0 or tcompact<=0:raise SystemExit('compact verification route proof failed')
if up>-1.0 or pp>0.5:raise SystemExit('performance gate failed')
PY
for d in 0 1 2 3;do grep -q "DFLASH_TP_LOCAL_VERIFY_TOP2_ACTIVE device=${d}" "${RESULTS}/top2.log";done
TM_DFLASH_TP_LOCAL_VERIFY_TOP2=1 python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1|tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS' "${RESULTS}/identity.log";touch "${RESULTS}/completed";echo DFLASH_TP_LOCAL_TOP2_COMPLETE
