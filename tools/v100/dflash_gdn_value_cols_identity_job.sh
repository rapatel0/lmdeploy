#!/usr/bin/env bash
# Resume the audited identity/qualification gates after a completed value-column matrix.
set -euo pipefail

RESULTS="${GDN_VALUE_COLS_RESULT_DIR:-}"
if [ -z "${RESULTS}" ]; then
    RESULTS="$(find /results -maxdepth 1 -type d -name '*-dflash-gdn-value-cols-*' | sort | tail -1)"
fi
[ -d "${RESULTS}" ] || { echo "FAIL: missing result directory ${RESULTS}" >&2; exit 2; }
exec > >(tee -a "${RESULTS}/identity_followup_console.log") 2>&1
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo 'FAIL: no wheel' >&2; exit 2; }
sha256sum "${WHEEL}" | tee "${RESULTS}/identity_wheel.sha256"
diff -u "${RESULTS}/wheel.sha256" "${RESULTS}/identity_wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
python3 /src/tools/v100/analyze_dflash_gdn_value_cols.py "${RESULTS}" | tee "${RESULTS}/analysis_followup.log"
WINNER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "${RESULTS}/analysis.json")"
[ -n "${WINNER}" ] || { echo DFLASH_GDN_VALUE_COLS_REJECTED_NO_WINNER; exit 4; }
WINNER_VALUE=${WINNER#v}

run_identity() {
    local value=$1 name=$2
    set +e
    TM_GDN_SM70_VALUE_COLS="${value}" python3 /job/verify_dflash_audited.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
        --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        2>&1 | tee "${RESULTS}/${name}.log"
    local rc=${PIPESTATUS[0]}
    set -e
    return "${rc}"
}

baseline_rc=0
winner_rc=0
run_identity 128 identity_v128_followup || baseline_rc=$?
run_identity "${WINNER_VALUE}" identity_winner_followup || winner_rc=$?

set +e
python3 - "${RESULTS}" "${WINNER}" "${baseline_rc}" "${winner_rc}" <<'PY'
import ast, json, re, sys
root, winner, baseline_rc, winner_rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
analysis = json.load(open(root + "/analysis.json", encoding="utf-8"))
def ids(path, label):
    text = open(path, encoding="utf-8", errors="replace").read()
    match = re.search(rf"^{label} token_ids=(\[.*\])$", text, re.M)
    return ast.literal_eval(match.group(1)) if match else None
baseline_k0 = ids(root + "/identity_v128_followup.log", "K=0")
winner_k0 = ids(root + "/identity_winner_followup.log", "K=0")
winner_k7 = ids(root + "/identity_winner_followup.log", "K=7")
identity = (baseline_rc == 0 and winner_rc == 0 and baseline_k0 is not None
            and baseline_k0 == winner_k0 == winner_k7)
result = {"status": "qualified" if identity else "rejected", "winner": winner,
          "identity": identity, "changes": analysis["changes"], "kernels": analysis["kernels"]}
open(root + "/qualification.json", "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_GDN_VALUE_COLS_" + result["status"].upper(), json.dumps(result, sort_keys=True))
raise SystemExit(0 if identity else 4)
PY
qualification_rc=$?
set -e
if [ "${qualification_rc}" -ne 0 ]; then
    echo DFLASH_GDN_VALUE_COLS_REJECTED
    exit "${qualification_rc}"
fi
rm -f "${RESULTS}/incomplete"
touch "${RESULTS}/completed"
echo DFLASH_GDN_VALUE_COLS_COMPLETE
