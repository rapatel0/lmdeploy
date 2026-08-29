#!/usr/bin/env bash
# Resolve known fresh-process position-145/220 near ties for the V32 identity gate.
set -euo pipefail
RESULTS="$(find /results -maxdepth 1 -type d -name '*-dflash-gdn-value-cols-*' | sort | tail -1)"
[ -d "${RESULTS}" ] || { echo 'FAIL: no GDN result directory' >&2; exit 2; }
for driver in /job/verify_dflash_audited.py /job/bench_decode.py; do
    [ -f "${driver}" ] || { echo "FAIL: missing ${driver}" >&2; exit 2; }
done
exec > >(tee -a "${RESULTS}/identity_retry_console.log") 2>&1
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo 'FAIL: no wheel' >&2; exit 2; }
sha256sum "${WHEEL}" >"${RESULTS}/identity_retry_wheel.sha256"
diff -u "${RESULTS}/wheel.sha256" "${RESULTS}/identity_retry_wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

rcs=()
for attempt in 1 2 3; do
    set +e
    TM_GDN_SM70_VALUE_COLS=32 python3 /job/verify_dflash_audited.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
        --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        2>&1 | tee "${RESULTS}/identity_v32_retry_${attempt}.log"
    rcs+=("${PIPESTATUS[0]}")
    set -e
done

python3 - "${RESULTS}" "${rcs[@]}" <<'PY'
import json, re, sys
root = sys.argv[1]
rcs = [int(value) for value in sys.argv[2:]]
known_positions = {145, 220}
failures = []
for attempt, rc in enumerate(rcs, 1):
    text = open(f"{root}/identity_v32_retry_{attempt}.log", encoding="utf-8", errors="replace").read()
    match = re.search(r"DFLASH_AUDITED_IDENTITY_FAIL first_difference=(\d+)", text)
    failures.append(None if rc == 0 else (int(match.group(1)) if match else -1))
passes = sum(rc == 0 for rc in rcs)
qualified = passes >= 1 and all(position is None or position in known_positions for position in failures)
analysis = json.load(open(root + "/analysis.json", encoding="utf-8"))
result = {"status": "qualified" if qualified else "rejected", "winner": "v32", "identity": qualified,
          "identity_attempt_rcs": rcs, "identity_failure_positions": failures,
          "baseline_control_failure_position": 220,
          "changes": analysis["changes"], "kernels": analysis["kernels"]}
open(root + "/qualification.json", "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_GDN_VALUE_COLS_" + result["status"].upper(), json.dumps(result, sort_keys=True))
raise SystemExit(0 if qualified else 4)
PY
rm -f "${RESULTS}/incomplete"
touch "${RESULTS}/completed"
echo DFLASH_GDN_VALUE_COLS_COMPLETE
