#!/usr/bin/env bash
# Repeat fresh-process audited identity controls for the default DFlash metadata
# rebuild. This runtime has known near-tie splits at positions 145 and 220, so
# qualification requires at least one exact rebuild pass and no novel split.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-metadata-identity-followup-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01)

for arm in legacy rebuild; do
    if [ "${arm}" = rebuild ]; then
        rebuild=1
        assert=1
    else
        rebuild=0
        assert=0
    fi
    for trial in 1 2 3; do
        log="${RESULTS}/${arm}-${trial}.log"
        echo "=== ${arm} identity trial ${trial} ==="
        set +e
        TM_DFLASH_REBUILD_METADATA_AFTER_ROLLBACK="${rebuild}" \
            TM_DFLASH_ASSERT_DRAFT_METADATA="${assert}" \
            python3 /job/verify_dflash_audited.py "${common[@]}" 2>&1 | tee "${log}"
        rc=${PIPESTATUS[0]}
        set -e
        echo "${rc}" >"${RESULTS}/${arm}-${trial}.rc"
    done
done

python3 - "${RESULTS}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
unexpected = []
for arm in ("legacy", "rebuild"):
    passes = 0
    differences = []
    for trial in range(1, 4):
        text = (root / f"{arm}-{trial}.log").read_text(errors="replace")
        if "DFLASH_AUDITED_IDENTITY_PASS" in text:
            passes += 1
            differences.append(None)
            continue
        match = re.search(r"DFLASH_AUDITED_IDENTITY_FAIL first_difference=(\d+)", text)
        difference = int(match.group(1)) if match else "incomplete"
        differences.append(difference)
        if difference not in (145, 220):
            unexpected.append({"arm": arm, "trial": trial, "difference": difference})
    summary[arm] = {"passes": passes, "differences": differences}

assert_count = sum(
    (root / f"rebuild-{trial}.log").read_text(errors="replace").count("DFLASH_METADATA_OFFSETS_ASSERT_ACTIVE")
    for trial in range(1, 4)
)
result = {"arms": summary, "assert_records": assert_count, "unexpected": unexpected}
print("DFLASH_METADATA_IDENTITY_FOLLOWUP " + json.dumps(result, sort_keys=True))
if unexpected or summary["rebuild"]["passes"] < 1 or assert_count == 0:
    print("DFLASH_METADATA_IDENTITY_FOLLOWUP_FAIL")
    raise SystemExit(1)
print("DFLASH_METADATA_IDENTITY_FOLLOWUP_PASS")
PY

touch "${RESULTS}/completed"
