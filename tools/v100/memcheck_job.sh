#!/usr/bin/env bash
# Where does speculative decoding touch memory it does not own?
#
# The K=4 run aborts with:
#   [TM][FATAL][kv_cache_utils_v2.cu:555] CUDA error: an illegal memory access
#
# Line 555 is a plain cudaGetLastError() at the end of invokeFlattenKV_v2. CUDA
# errors are asynchronous, so that names where the error was DETECTED, not where
# it happened -- any kernel launched earlier on the stream can be the source.
# Reading that file as the location cost many hours.
#
# compute-sanitizer answers the question directly: it names the faulting kernel,
# the address, the access size, and the allocation the address fell outside of.
# This job runs the smallest workload that still reaches the fault, under
# memcheck, and keeps the report.
#
# It deliberately does NOT run the K=0 baseline or the throughput benchmarks.
# The baseline is already known to be clean, and memcheck costs 10-100x runtime,
# so every extra token is sanitizer time spent confirming something known.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-memcheck-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1

# Mark the run incomplete until it reaches its own end.
#
# The EXIT trap records whatever status the shell had when it died, so a job
# killed mid-build writes exit_code 0 and is indistinguishable from a clean
# pass. `completed` is written only on the last line, so exit_code 0 AND
# completed present is the only combination that means anything.
finish() {
    rc=$?
    echo "$rc" >"${RESULTS}/exit_code"
    if [ ! -f "${RESULTS}/completed" ]; then
        echo "KILLED_BEFORE_COMPLETION" >"${RESULTS}/incomplete"
        echo "WARNING: job did not reach its end; exit ${rc} is not a verdict" >&2
    fi
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

echo "=== source ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "no SOURCE_STAMP"

echo
echo "=== locate compute-sanitizer ==="
# Prefer PATH, fall back to the toolkit location. Without this the job would
# fail deep inside the run with "command not found" after a 25-minute build.
CS="$(command -v compute-sanitizer 2>/dev/null)"
[ -n "${CS}" ] || CS=/usr/local/cuda/bin/compute-sanitizer
if [ ! -x "${CS}" ]; then
    echo "FAIL: compute-sanitizer not found (tried PATH and /usr/local/cuda/bin)" >&2
    exit 2
fi
echo "  ${CS}"
"${CS}" --version 2>&1 | head -3

echo
echo "=== build ==="
# The progress line stalls at roughly 351/359 for several minutes while
# kv_cache_utils_v2.cu compiles. That is not a hang.
cd /src
if ! bash /src/tools/v100/build_v100.sh >"${RESULTS}/build.log" 2>&1; then
    echo "FAIL: build" >&2
    echo "--- compiler diagnostics ---"
    grep -aE "error:|Error [0-9]+|undefined reference|no member named|no matching" \
        "${RESULTS}/build.log" | head -40
    echo "--- tail of build log ---"
    tail -25 "${RESULTS}/build.log"
    exit 2
fi
tail -3 "${RESULTS}/build.log"

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo "FAIL: no wheel" >&2
    exit 2
}
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
TP="${TP:-4}"
K="${NUM_DRAFT_TOKENS:-4}"
MAXNEW="${MAX_NEW_TOKENS:-32}"
cd /

echo
echo "=== configuration ==="
echo "  MODEL=${MODEL} TP=${TP} K=${K} max_new=${MAXNEW}"
[ "${K}" -gt 0 ] || {
    echo "FAIL: K=${K} would run the clean baseline, which has nothing to find" >&2
    exit 3
}

# Stage A: CUDA_LAUNCH_BLOCKING at full TP, no sanitizer.
#
# The first attempt ran memcheck against all four TP ranks and deadlocked:
# one GPU pinned at 100%, three at 0%, no log output for 53 minutes. The
# sanitizer serialises kernel launches per process, and NCCL collectives
# across instrumented ranks degrade to a lockstep that never converged.
#
# Launch-blocking has no such interaction. Every launch becomes synchronous,
# so the abort names the true launch site instead of wherever the async error
# happened to surface. No address, no allocation -- but the correct kernel,
# at normal execution speed, at the exact TP=4 configuration that faults.
echo
echo "=== stage A: CUDA_LAUNCH_BLOCKING=1, TP=${TP}, no sanitizer ==="
BLOCKING_LOG="${RESULTS}/launch_blocking.log"
CUDA_LAUNCH_BLOCKING=1 TM_SPEC_VALIDATE_IDS=1 TM_DEBUG_LEVEL=INFO \
    timeout -k 30 900 stdbuf -oL -eL \
    python3 /src/tools/v100/memcheck_child.py \
    --model-dir "${MODEL}" --tp "${TP}" \
    --num-draft-tokens "${K}" --max-new-tokens "${MAXNEW}" \
    2>&1 | tee "${BLOCKING_LOG}"
A_RC="${PIPESTATUS[0]}"
echo "  stage A exit ${A_RC} (124 = timeout)"

echo
echo "=== stage A findings ==="
grep -aE "CUDA error|FATAL|what\(\)|Aborted" "${BLOCKING_LOG}" | head -10 ||
    echo "  no fault reported"

# Stage B: memcheck at TP=2.
#
# TP=1 cannot hold the model: the weights alone are 29G against 32G of HBM.
# TP=2 halves the rank count and the collective fan-in that hung TP=4. It may
# still hang -- so it runs under a hard timeout, and stage A's answer is
# already on disk before this starts. If the fault does not reproduce at TP=2,
# that is itself a finding (points at cross-rank state), reported honestly
# below rather than as a clean bill.
echo
echo "=== stage B: memcheck, TP=2, hard timeout 40m ==="
MEMCHECK_LOG="${RESULTS}/memcheck.log"
timeout -k 30 2400 stdbuf -oL -eL "${CS}" \
    --tool memcheck \
    --launch-timeout 120 \
    --target-processes all \
    --error-exitcode 88 \
    --print-limit 20 \
    python3 /src/tools/v100/memcheck_child.py \
    --model-dir "${MODEL}" --tp 2 \
    --num-draft-tokens "${K}" --max-new-tokens "${MAXNEW}" \
    2>&1 | tee "${MEMCHECK_LOG}"
MC_RC="${PIPESTATUS[0]}"
if [ "${MC_RC}" = "124" ] || [ "${MC_RC}" = "137" ]; then
    echo "  stage B TIMED OUT under the sanitizer -- treat its log as partial"
fi

echo
echo "=== sanitizer findings ==="
echo "  compute-sanitizer exit ${MC_RC}"

# Count the errors the sanitizer itself reports, rather than inferring from the
# exit status. A run that dies for an unrelated reason must not read as "no
# memory errors found".
ERRS="$(grep -acE "^=+ *(Invalid|Program hit)" "${MEMCHECK_LOG}" 2>/dev/null || echo 0)"
SUMMARY="$(grep -aE "ERROR SUMMARY" "${MEMCHECK_LOG}" 2>/dev/null | tail -1)"

echo "  error stanzas: ${ERRS}"
[ -n "${SUMMARY}" ] && echo "  ${SUMMARY}"

if [ "${ERRS}" -gt 0 ]; then
    echo
    echo "--- first faulting kernel and address ---"
    # The stanza header carries the access kind and size; the following lines
    # carry the kernel name and the nearest allocation.
    grep -aE "Invalid .* of size|Address 0x[0-9a-f]+ is|at 0x[0-9a-f]+ in |Saved host backtrace|by thread" \
        "${MEMCHECK_LOG}" | head -30

    echo
    echo "--- kernel names implicated ---"
    grep -aoE "in [A-Za-z_:<>0-9, ]+\(" "${MEMCHECK_LOG}" | sort | uniq -c |
        sort -rn | head -15
fi

echo
echo "=== verdict ==="
# Stage A is the primary result: it runs the exact faulting configuration.
# Stage B is supplementary: richer detail, different configuration.
echo "--- stage A (launch-blocking, TP=${TP}) ---"
if grep -aqE "CUDA error|FATAL" "${BLOCKING_LOG}"; then
    echo "STAGE_A_FAULTED: the lines above name the true launch site"
    grep -aE "\[TM\]\[FATAL\]" "${BLOCKING_LOG}" | head -3
elif [ "${A_RC}" = "124" ]; then
    echo "STAGE_A_TIMEOUT: did not fault within 15m under launch-blocking"
else
    echo "STAGE_A_CLEAN: ran to completion with no fault (exit ${A_RC})"
    echo "  a fault that vanishes under synchronous launches points at an"
    echo "  ordering race, not a fixed bad address"
fi

echo "--- stage B (memcheck, TP=2) ---"
if [ "${ERRS}" -gt 0 ]; then
    echo "MEMCHECK_FOUND_ERRORS: ${ERRS} stanza(s) -- see ${MEMCHECK_LOG}"
elif [ "${MC_RC}" = "124" ] || [ "${MC_RC}" = "137" ]; then
    echo "MEMCHECK_TIMEOUT: no verdict from stage B; rely on stage A"
elif [ -z "${SUMMARY}" ]; then
    echo "MEMCHECK_NO_REPORT: sanitizer never attached; rely on stage A"
else
    echo "MEMCHECK_CLEAN at TP=2: fault did not reproduce, or needs TP=4"
fi

touch "${RESULTS}/completed"
echo "MEMCHECK_COMPLETE"
