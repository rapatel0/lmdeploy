#!/usr/bin/env bash
# One-build proof that rejected verifier target-KV suffixes are invisible.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-rejected-kv-poison-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
started=0
if [ "${REUSE_WHEEL:-0}" != 1 ]; then
  rm -f /wheels/lmdeploy-*.whl
  started=$(date +%s)
  bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || { tail -100 "${RESULTS}/build.log"; exit 2; }
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
if [ "${REUSE_WHEEL:-0}" != 1 ]; then [ "$(stat -c %Y "${WHEEL}")" -ge "${started}" ]; fi
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
  --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
  --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256
  --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
  --cache-max-entry-count 0.05)
for arm in control_a poison control_b; do
  poison=0
  if [ "${arm}" = poison ]; then
    poison=1
  fi
  TM_DFLASH_KV_PROPOSAL_TRACE=1 TM_DFLASH_POISON_REJECTED_KV="${poison}" \
    python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/${arm}.json" \
    2>&1 | tee "${RESULTS}/${arm}.log"
  python3 - "${RESULTS}/${arm}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert len(d['trials'])==5
assert all(not x.get('degenerate') and x.get('output_tokens')==256 for x in d['trials'])
PY
done
for device in 0 1 2 3; do
  grep -q "DFLASH_REJECTED_KV_POISON_ACTIVE device=${device} " "${RESULTS}/poison.log"
done
if grep -q 'DFLASH_REJECTED_KV_POISON_ACTIVE' "${RESULTS}/control_a.log"; then
  echo 'control unexpectedly poisoned target KV' >&2
  exit 3
fi
python3 /job/analyze_dflash_rejected_kv_poison.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
TM_DFLASH_POISON_REJECTED_KV=1 python3 /job/verify_dflash_audited.py \
  --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
  --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
  --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
  2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo DFLASH_REJECTED_KV_POISON_COMPLETE
