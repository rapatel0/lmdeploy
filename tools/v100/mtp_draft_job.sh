#!/usr/bin/env bash
# Prove the MTP draft path executes. Assumes the wheel is already built.
#
# The driver deliberately does not capture the engine's stderr: the engine
# aborts inside C++, which kills the process with no exception and no chance to
# copy a captured buffer back out. Redirecting stderr hid the abort message
# twice. So the whole run is tee'd here, and the "[MTP] drafted" records are
# read back from that transcript afterwards.
set -uo pipefail

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "${WHEEL}" ]; then
    echo "FAIL: no wheel in /wheels; run the build job first" >&2
    exit 2
fi
echo "=== installing $(basename "${WHEEL}") ==="
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
[ -f "${MODEL}/config.json" ] || { echo "FAIL: no checkpoint at ${MODEL}" >&2; exit 3; }

LOG=/tmp/mtp-run.log
echo
echo "=== run the draft path ==="
cd /
# stdbuf so the section markers interleave correctly with the C++ records.
stdbuf -oL -eL python3 /src/tools/v100/verify_mtp_draft.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}

echo
echo "=== the C++ draft records, read back from the transcript ==="
if [ "${rc}" -ne 0 ]; then
    echo "run failed (rc=${rc}); the abort message is above" >&2
    exit "${rc}"
fi

# Claim 1: with drafting off, no draft may occur.
if sed -n '/1. baseline/,/2. drafting on/p' "${LOG}" | grep -q '\[MTP\] drafted'; then
    echo "FAIL: a draft ran with num_draft_tokens=0; the guard is not honoured" >&2
    exit 4
fi

# Claim 2: with drafting on, at least one draft must occur.
DRAFTS="$(sed -n '/2. drafting on/,$p' "${LOG}" | grep -c '\[MTP\] drafted')"
sed -n '/2. drafting on/,$p' "${LOG}" | grep '\[MTP\] drafted' | head -4
if [ "${DRAFTS}" -eq 0 ]; then
    echo "FAIL: no draft record. The path is dead code: mtp_predictor_ is null," >&2
    echo "or num_draft_tokens never reached the engine." >&2
    exit 5
fi
echo "  ${DRAFTS} draft record(s)"

echo
echo "VERIFY_MTP_DRAFT_PASS"
