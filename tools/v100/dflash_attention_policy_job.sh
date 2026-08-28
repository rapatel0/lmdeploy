#!/usr/bin/env bash
# Measure the published draft checkpoint's sliding-decoder attention contract.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-attention-policy-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO
export TM_DFLASH_CONTEXT_BF16_ROUND=0
export TM_MTP_AMBIGUITY_MARGIN=0
export TM_DFLASH_REDUCE_BEFORE_CONV=0

for legacy in 1 0; do
    arm="legacy_policy${legacy}"
    echo "=== ATTENTION_POLICY_ARM ${arm} ==="
    TM_DFLASH_LEGACY_ATTENTION_POLICY="${legacy}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 2 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "${RESULTS}/${arm}.json" \
        2>&1 | tee "${RESULTS}/${arm}.log"
done

# The corrected default must preserve exact output on the same audited prompt.
TM_DFLASH_LEGACY_ATTENTION_POLICY=0 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/corrected_identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/corrected_identity.log"

python3 - "${RESULTS}" <<'PY'
import json
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"final commit length ([0-9.]+), raw ([0-9.]+)")
rows = []
for result in sorted(root.glob("legacy_policy*.json")):
    data = json.loads(result.read_text())
    matches = pattern.findall((root / f"{result.stem}.log").read_text(errors="replace"))
    final, raw = matches[-1] if matches else (None, None)
    rows.append({
        "arm": result.stem,
        "decode_tok_s": data.get("mean_decode_tok_s"),
        "commit_length": None if final is None else float(final),
        "raw_commit_length": None if raw is None else float(raw),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

touch "${RESULTS}/completed"
echo DFLASH_ATTENTION_POLICY_COMPLETE
