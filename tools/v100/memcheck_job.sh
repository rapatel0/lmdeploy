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
[ "${TP}" = "4" ] || {
    echo "FAIL: island is 4 GPUs, TP=${TP} invalid" >&2
    exit 3
}
[ "${K}" -gt 0 ] || {
    echo "FAIL: K=${K} would run the clean baseline, which has nothing to find" >&2
    exit 3
}

echo
echo "=== memcheck: K=${K} generation ==="
# --target-processes all is essential, not optional. The engine runs generation
# in a child process, so without it the sanitizer would watch the parent, see no
# kernels, and report a clean run for a job that crashed.
#
# --launch-timeout is raised because memcheck serialises launches and the
# default is easily exceeded on a 27B model across 4 GPUs.
#
# No CUDA_LAUNCH_BLOCKING here: memcheck already attributes the fault to its
# kernel, and blocking would multiply an already slow run.
MEMCHECK_LOG="${RESULTS}/memcheck.log"
stdbuf -oL -eL "${CS}" \
    --tool memcheck \
    --launch-timeout 120 \
    --target-processes all \
    --error-exitcode 88 \
    --print-limit 20 \
    python3 /src/tools/v100/memcheck_child.py \
    --model-dir "${MODEL}" --tp "${TP}" \
    --num-draft-tokens "${K}" --max-new-tokens "${MAXNEW}" \
    2>&1 | tee "${MEMCHECK_LOG}"
MC_RC="${PIPESTATUS[0]}"

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
if [ "${ERRS}" -eq 0 ] && [ -z "${SUMMARY}" ]; then
    echo "FAIL: the sanitizer produced no report at all; it likely never attached" >&2
    echo "      check ${MEMCHECK_LOG} for a launch failure" >&2
    exit 9
fi

if [ "${ERRS}" -gt 0 ]; then
    echo "MEMCHECK_FOUND_ERRORS: ${ERRS} stanza(s) -- see ${MEMCHECK_LOG}"
else
    # A clean run under memcheck is itself a finding: it means the fault does
    # not reproduce under serialised launches, which points at a race rather
    # than a fixed out-of-bounds address.
    echo "MEMCHECK_CLEAN: no memory errors under this workload"
    echo "  either the workload is too small to reach the fault, or the fault"
    echo "  is a race that serialised execution hides"
fi

touch "${RESULTS}/completed"
echo "MEMCHECK_COMPLETE"
