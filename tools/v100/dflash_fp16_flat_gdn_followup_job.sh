#!/usr/bin/env bash
# No-build counter-ordered confirmation for the flat native FP16 GDN arm.
set -euo pipefail
BASE=/results/20260829_135435-dflash-fp16-flat-gdn-31d8741be859
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-flat-gdn-followup
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
[ -f "${BASE}/baseline.json" ] && [ -f "${BASE}/native.json" ] || { echo 'FAIL: original matrix unavailable'; exit 2; }
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo 'FAIL: rebuilt wheel unavailable'; exit 2; }
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
        --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256
        --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
        --cache-max-entry-count 0.05)
# Counter-order the original baseline->transposed->native matrix.
for arm in native baseline; do
    mode=2; [ "${arm}" = baseline ] && mode=0
    TM_SM70_FP16_FLAT_GDN="${mode}" python3 /job/bench_decode.py "${common[@]}" --trials 5 \
        --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
done
python3 - "${BASE}" "${RESULTS}" <<'PY'
import json,re,sys
from pathlib import Path
base,out=map(Path,sys.argv[1:])
pat=re.compile(r'final commit length [0-9.]+, raw [0-9.]+ over (\d+) verification steps \((\d+) committed,')
report={}
for arm in ('baseline','native'):
    decode_means=[]; committed=steps=0
    for root in (base,out):
        data=json.loads((root/f'{arm}.json').read_text())
        rows=data['trials']; assert len(rows)==5 and all(not r.get('degenerate') for r in rows)
        decode_means.append(float(data['mean_decode_tok_s']))
        matches=pat.findall((root/f'{arm}.log').read_text(errors='replace'))
        assert matches
        step,count=map(int,matches[-1]); steps+=step; committed+=count
    decode=sum(decode_means)/len(decode_means)
    commit=committed/steps
    report[arm]={'decode_tok_s':decode,'commit_length':commit,'cycle_ms':commit/decode*1000,
                 'trials':10,'committed':committed,'verification_steps':steps}
gain=(1-report['native']['cycle_ms']/report['baseline']['cycle_ms'])*100
report['cycle_gain_pct']=gain; report['qualified']=gain>=1.0
(out/'followup.json').write_text(json.dumps(report,indent=2)+'\n')
print('DFLASH_FP16_FLAT_GDN_FOLLOWUP',json.dumps(report,sort_keys=True))
raise SystemExit(0 if report['qualified'] else 4)
PY
touch "${RESULTS}/completed"
echo DFLASH_FP16_FLAT_GDN_FOLLOWUP_PASS
