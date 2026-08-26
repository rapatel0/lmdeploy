#!/usr/bin/env bash
# Prove the MTP draft path executes. Assumes the wheel is already built.
#
# The driver deliberately does not capture the engine's stderr: the engine
# aborts inside C++, which kills the process with no exception and no chance to
# copy a captured buffer back out. Redirecting stderr hid the abort message
# twice. So the whole run is tee'd here, and the "[MTP] drafted" records are
# read back from that transcript afterwards.
set -uo pipefail

# Persist everything to the /results hostPath, not just pod-local /tmp. The
# pod is reclaimed when the job finishes and its logs go with it, so anything
# only in /tmp cannot be read afterwards.
# Stamp the directory with the source commit actually under test, read from
# /src rather than passed in, so the name cannot claim a commit the build did
# not use.
#
# Read SOURCE_STAMP, not git. sync_src.sh ships the tree with `git archive`,
# which deliberately carries no .git directory, so `git rev-parse` here finds
# no repository and quietly yields "unknown" -- which is what the first two
# result directories are named. The stamp is the mechanism that already exists
# for exactly this question.
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
echo "results dir: ${RESULTS}"

finish() {
    rc=$?
    for f in /tmp/mtp-base.log /tmp/mtp-spec.log /tmp/base.txt /tmp/spec.txt; do
        [ -e "$f" ] && cp -f "$f" "${RESULTS}/" 2>/dev/null
    done
    echo "$rc" >"${RESULTS}/exit_code"
    echo "artifacts saved to ${RESULTS} (exit ${rc})"
}
trap finish EXIT

# Build, always. The wheel carries the Python sources too, so installing a
# wheel that is older than /src silently reverts any Python fix and the run
# then tests stale code. That already happened once: the fc fix was in /src
# and absent from the wheel, and the identical abort looked like the fix had
# failed. Ninja makes this cheap when no C++ changed.
echo "=== build ==="
cd /src
bash /src/tools/v100/build_v100.sh || {
    echo "FAIL: build" >&2
    exit 2
}

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo "FAIL: no wheel after build" >&2
    exit 2
}
echo "=== installing $(basename "${WHEEL}") ==="
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -2

# The wheel must actually contain the fix that is in /src.
#
# Read the wheel itself, not an import. cwd is /src from the build above, so
# `import lmdeploy` resolves to the source tree and would report the fix
# present no matter what the wheel holds -- the same class of false green
# signal that let a stale wheel masquerade as a tested fix.
python3 - "${WHEEL}" <<'CHECK' || exit 2
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    blob = z.read("lmdeploy/turbomind/builders/mtp_layer.py").decode()
if "add_fc" not in blob:
    raise SystemExit("FAIL: the built wheel predates the fc fix")
print("  the wheel's mtp_layer.py carries add_fc")
CHECK

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
[ -f "${MODEL}/config.json" ] || {
    echo "FAIL: no checkpoint at ${MODEL}" >&2
    exit 3
}

echo
echo "=== run the draft path ==="
cd /
# Two separate processes, one generation each. A 27B model at tp=4 fills most
# of each 32GB V100, and nothing frees the first pipeline's weights, so doing
# both in one process OOMs on the second load.
BASE_LOG=/tmp/mtp-base.log
SPEC_LOG=/tmp/mtp-spec.log

echo "=== 1. baseline, drafting off ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_mtp_draft.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 0 --max-new-tokens 256 \
    --emit-text /tmp/base.txt 2>&1 | tee "${BASE_LOG}"
# Judge by the artifact, not the exit code. TurboMind can abort during
# ~Impl() with "Resource deadlock avoided" AFTER generation has completed and
# the text has been written -- upstream teardown (#4770 territory), reachable
# with speculation switched off and in code I have never touched. Treat a
# written, non-empty result as success and say so, rather than discarding a
# good generation because the process died on its way out.
if [ ! -s /tmp/base.txt ]; then
    echo "FAIL: baseline produced no text" >&2
    exit 4
fi
BASE_RC="${PIPESTATUS[0]}"
if ! grep -q "EMIT_TEXT_COMPLETE" "${BASE_LOG}"; then
    echo "FAIL: baseline did not reach the end of main(); rc=${BASE_RC}" >&2
    exit 8
fi
if [ "${BASE_RC}" -ne 0 ]; then
    # Report the actual status. 134 is SIGABRT, 139 SIGSEGV, 1 an ordinary
    # non-zero return; calling all of them "teardown" hid which one this is.
    echo "  NOTE: baseline generated text, then exited rc=${BASE_RC} after the text was written"
fi

echo
echo "=== 2. drafting on, depth 4 ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_mtp_draft.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4 --max-new-tokens 256 \
    --emit-text /tmp/spec.txt 2>&1 | tee "${SPEC_LOG}"
if [ ! -s /tmp/spec.txt ]; then
    echo "FAIL: speculative run produced no text" >&2
    exit 5
fi
SPEC_RC="${PIPESTATUS[0]}"
if ! grep -q "EMIT_TEXT_COMPLETE" "${SPEC_LOG}"; then
    echo "FAIL: speculative run did not reach the end of main(); rc=${SPEC_RC}" >&2
    exit 8
fi
if [ "${SPEC_RC}" -ne 0 ]; then
    echo "  NOTE: speculative run generated text, then exited rc=${SPEC_RC} after the text was written"
fi

echo
echo "=== 3. did a draft actually run? ==="
# Claim 1: with drafting off, no draft may occur.
if grep -q '\[MTP\] drafted' "${BASE_LOG}"; then
    echo "FAIL: a draft ran with num_draft_tokens=0; the guard is not honoured" >&2
    exit 6
fi
# Claim 2: with drafting on, at least one draft must occur.
DRAFTS="$(grep -c '\[MTP\] drafted' "${SPEC_LOG}")"
grep '\[MTP\] drafted' "${SPEC_LOG}" | head -4
if [ "${DRAFTS}" -eq 0 ]; then
    echo "FAIL: no draft record. The path is dead code: mtp_predictor_ is null," >&2
    echo "or num_draft_tokens never reached the engine." >&2
    exit 7
fi
echo "  ${DRAFTS} draft record(s)"

echo
echo "=== 3b. step-1 draft acceptance ==="
# The number this whole exercise exists to produce. Near-zero is the EXPECTED
# result right now: the MTP KV slot is registered but never written, so the
# draft attends to uninitialised memory. Recording it before the seeding fix is
# the point -- afterwards, a fix that changes nothing is indistinguishable from
# a fix that works.
if grep -aq "step-1 draft acceptance" "${SPEC_LOG}"; then
    grep -a "step-1 draft acceptance" "${SPEC_LOG}" | tail -3
else
    echo "  no acceptance report: fewer than 32 scored draft/token pairs"
fi

echo "=== 4. is the output unchanged? ==="
# Drafts are discarded today, so speculation must be invisible in the text.
# A difference here means the draft is corrupting the target's KV or
# recurrent state -- which would not raise, only degrade.
if ! diff -q /tmp/base.txt /tmp/spec.txt >/dev/null; then
    echo "FAIL: drafting altered the output." >&2
    echo "--- without drafting ---" >&2
    head -c 400 /tmp/base.txt >&2
    echo >&2
    echo "--- with drafting ---" >&2
    head -c 400 /tmp/spec.txt >&2
    echo >&2
    exit 8
fi
echo "  identical, as required while drafts are discarded"

echo
echo "VERIFY_MTP_DRAFT_PASS"

# --- probe: does the absolute base position matter at all? ---
#
# A uniform +1 shift of every drafted position left steps 1-3 bit-identical.
# The shift is NEGATIVE: forward slack is often under 32, so +32 crashed with
# an illegal access by walking past the row block. Backward stays inside memory
# the row already owns.
# That is only possible if the entries below the base contribute something
# invariant to the shift. Zeros do; leftover bytes do not. Shifting by a large
# constant discriminates in one run.
#
# This deliberately ignores block slack, so its OUTPUT IS NOT TRUSTED -- only
# the acceptance table is read, and only to compare against the run above.
echo
echo "=== 4. probe: base shifted by -8 (output deliberately untrusted) ==="
TM_MTP_PROBE_SHIFT=-8 stdbuf -oL -eL python3 /src/tools/v100/verify_mtp_draft.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4 --max-new-tokens 64 \
    --emit-text /tmp/probe.txt 2>&1 | tail -25 || true

# --- the goal's actual question: is inference working, stable, appropriate? ---
#
# Everything above compares speculative text against baseline text. Two
# identical WRONG answers pass that test. This checks correctness against known
# answers, across several prompts and lengths, many generations in one process,
# and asserts determinism under greedy decoding.
echo
echo "=== 5. inference correctness and stability (drafting off) ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_inference.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 0 --rounds 3 2>&1
INFER_BASE_RC="${PIPESTATUS[0]}"

echo
echo "=== 6. inference correctness and stability (drafting on, depth 4) ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_inference.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4 --rounds 3 2>&1
INFER_SPEC_RC="${PIPESTATUS[0]}"

echo
echo "=== inference verdict ==="
echo "  drafting off rc=${INFER_BASE_RC}, drafting on rc=${INFER_SPEC_RC}"

# --- concurrency and long output: the cases bsz==1 never reaches ---
#
# The MTP draft path takes the MINIMUM block slack across the batch, so with a
# single row it is just that row's slack. Mixed lengths in one forward exercise
# a path no run so far has touched. Long generation crosses many block
# boundaries in one sequence.
echo
echo "=== 7. concurrency + long output (drafting off) ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_concurrent.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 0 --repeats 2 2>&1
CONC_BASE_RC="${PIPESTATUS[0]}"

echo
echo "=== 8. concurrency + long output (drafting on, depth 4) ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_concurrent.py \
    --model-dir "${MODEL}" --tp 4 --num-draft-tokens 4 --repeats 2 2>&1
CONC_SPEC_RC="${PIPESTATUS[0]}"

echo
echo "=== concurrency verdict ==="
echo "  drafting off rc=${CONC_BASE_RC}, drafting on rc=${CONC_SPEC_RC}"

# Prove the batch really was concurrent rather than five sequential requests.
#
# Passing a list to the pipeline is not evidence: Pipeline._infer creates a
# task per prompt under a semaphore of max_batch_size, but a scheduler that
# admitted them one at a time would still return five correct answers and the
# test would pass having measured nothing.
#
# The scheduler logs one line per admitted request carrying its uid. More than
# one DISTINCT uid inside the concurrency sections means rows genuinely shared
# a forward -- which is the shape MTP's min-slack-across-the-batch has never
# been exercised in.
UIDS="$(grep -aoE "req [0-9]+ \(uid [0-9]+\)" "${RESULTS}/console.log" 2>/dev/null |
    sort -u | wc -l | tr -d " ")"
echo "  distinct scheduler uids observed: ${UIDS}"
if [ "${UIDS}" -lt 2 ]; then
    echo "  WARNING: never saw two requests admitted; concurrency was NOT exercised"
fi

# Make the job's exit code mean something.
#
# Sections 5-8 captured four return codes and only PRINTED them. The job would
# have exited 0 with every correctness and concurrency check failing, and I was
# about to read that 0 as proof. This is the same defect as the codex wrapper
# that "succeeded" while running nothing -- a green result produced by a gate
# that was never wired up.
FAILED=0
for pair in "inference/drafting-off:${INFER_BASE_RC}" \
            "inference/drafting-on:${INFER_SPEC_RC}" \
            "concurrency/drafting-off:${CONC_BASE_RC}" \
            "concurrency/drafting-on:${CONC_SPEC_RC}"; do
    name="${pair%%:*}"
    rc="${pair##*:}"
    if [ "${rc}" -ne 0 ]; then
        echo "FAIL: ${name} returned rc=${rc}" >&2
        FAILED=1
    fi
done
if [ "${FAILED}" -ne 0 ]; then
    echo "FAIL: one or more verification sections failed" >&2
    exit 9
fi
echo "ALL_VERIFICATION_SECTIONS_PASS"
