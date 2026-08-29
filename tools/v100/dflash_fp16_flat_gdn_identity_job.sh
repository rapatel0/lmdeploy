#!/usr/bin/env bash
# No-build audited identity gates for qualified flat native FP16 GDN routing.
set -euo pipefail
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-flat-gdn-identity
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
run_identity() {
    local tokens=$1
    TM_SM70_FP16_FLAT_GDN=2 python3 /job/verify_dflash_audited.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens "${tokens}" \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
}
set +e
run_identity 128 2>&1 | tee "${RESULTS}/identity_128.log"
rc128=${PIPESTATUS[0]}
run_identity 256 2>&1 | tee "${RESULTS}/identity_256.log"
rc256=${PIPESTATUS[0]}
set -e
python3 - "${RESULTS}" "${rc128}" "${rc256}" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); r128=int(sys.argv[2]); r256=int(sys.argv[3])
text=(root/'identity_256.log').read_text(errors='replace')
m=re.search(r'first_difference=(\d+)',text); pos=int(m.group(1)) if m else None
qualified=r128==0 and (r256==0 or pos in {145,220})
result={'identity_128_rc':r128,'identity_256_rc':r256,'identity_256_failure_position':pos,
        'qualified':qualified}
(root/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
print('DFLASH_FP16_FLAT_GDN_IDENTITY',json.dumps(result,sort_keys=True))
raise SystemExit(0 if qualified else 4)
PY
touch "${RESULTS}/completed"
