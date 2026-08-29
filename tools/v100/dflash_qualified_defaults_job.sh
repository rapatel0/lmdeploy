#!/usr/bin/env bash
# Rebuild and smoke the stacked default-on DFlash optimizations.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-qualified-defaults-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
rm -f /wheels/lmdeploy-*.whl
started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
 tail -100 "${RESULTS}/build.log"
 exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ "$(stat -c %Y "${WHEEL}")" -ge "${started}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256 --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 --cache-max-entry-count 0.05)
python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/default.json" 2>&1 | tee "${RESULTS}/default.log"
python3 - "${RESULTS}" <<'PY'
import json,re,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]); text=(r/'default.log').read_text(errors='replace'); d=json.loads((r/'default.json').read_text())
assert len(d['trials'])==5 and all(not x.get('degenerate') and x.get('output_tokens')==256 for x in d['trials'])
for dev in range(4):
 for marker in (f'SM70_FP16_FLAT_GDN_ACTIVE device={dev} mode=2 shape=5120x4120',f'SM70_FP16_FLAT_HEAD_ACTIVE device={dev} mode=1 shape=5120x62080',f'DFLASH_TP_LOCAL_VERIFY_TOP2_ACTIVE device={dev} rows=8 local_vocab=62080'):
  if text.count(marker)!=1: raise SystemExit(f'missing default marker: {marker}')
m=re.findall(r'final commit length ([0-9.]+), raw [0-9.]+ over (\d+) verification steps',text)
if not m: raise SystemExit('missing acceptance')
commit=statistics.median(float(x[0]) for x in m); decode=float(d['mean_decode_tok_s']); cycle=1000*commit/decode
out={'decode_tok_s':decode,'commit_length':commit,'normalized_cycle_ms':cycle,'trials':len(d['trials'])};(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print('DFLASH_QUALIFIED_DEFAULTS_ROUTE_PASS',json.dumps(out,sort_keys=True))
PY
python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo DFLASH_QUALIFIED_DEFAULTS_COMPLETE
