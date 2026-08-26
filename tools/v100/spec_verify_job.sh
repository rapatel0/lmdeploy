#!/usr/bin/env bash
# Does speculative decoding produce IDENTICAL output and go FASTER?
#
# Both halves matter and neither is optional.
#
# Greedy speculative decoding is an exactness-preserving optimisation: the
# target model verifies every drafted token, so accepting them must not change
# a single output token. If the text differs, the verification is wrong and any
# speedup is meaningless -- it is a different, cheaper computation, not the same
# one done faster.
#
# So this runs the same prompts at K=0 and K=4 and requires byte-identical text
# before it will even look at the timing.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-specverify-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1

finish() {
    rc=$?
    echo "$rc" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

echo "=== source ==="
cat /src/SOURCE_STAMP 2>/dev/null || echo "no SOURCE_STAMP"

echo
echo "=== build ==="
cd /src
# Keep the FULL build log.
#
# `| tail -5` discarded the compiler diagnostic and left only cmake's
# "returned non-zero exit status 1", which reports that a build failed but not
# why. A one-line compile error then costs an entire GPU job to observe.
#
# The pipe was also swallowing the exit status: `cmd | tail` reports tail's
# status, so only `set -o pipefail` made that `||` fire at all.
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
cd /

echo
echo "=== configuration ==="
echo "  MODEL=${MODEL} TP=${TP} K=${K}"
[ "${TP}" = "4" ] || {
    echo "FAIL: island is 4 GPUs, TP=${TP} invalid" >&2
    exit 3
}

echo
echo "=== 1. correctness: K=0 vs K=${K}, output must be identical ==="
stdbuf -oL -eL python3 /src/tools/v100/verify_spec_identity.py \
    --model-dir "${MODEL}" --tp "${TP}" --num-draft-tokens "${K}" \
    --json-out "${RESULTS}/identity.json" 2>&1
IDENT_RC=$?

echo
echo "=== 2. accept length reported by the engine ==="
ACC="$(grep -aoE '\[MTP\] accept length [0-9.]+ tokens/step over [0-9]+ steps' "${RESULTS}/console.log" | tail -3)"
if [ -z "${ACC}" ]; then
    echo "WARN: no accept-length line; drafting may not have engaged"
    ACC_SEEN=0
else
    echo "${ACC}"
    ACC_SEEN=1
fi

echo
echo "=== 3. throughput: K=0 vs K=${K} ==="
for kk in 0 "${K}"; do
    echo "--- num_draft_tokens=${kk}"
    stdbuf -oL -eL python3 /src/tools/v100/bench_decode.py \
        --model "${MODEL}" --tp "${TP}" --num-draft-tokens "${kk}" \
        --input-tokens 1024 --output-tokens 256 --trials 3 \
        --require-mtp --json-out "${RESULTS}/bench_k${kk}.json" 2>&1
    eval "BENCH_RC_${kk}=\$?"
done

echo
echo "=== comparison ==="
python3 - "${RESULTS}/bench_k0.json" "${RESULTS}/bench_k${K}.json" <<'PY'
import json, sys

def load(p):
    try:
        with open(p) as h:
            return json.load(h)
    except Exception as exc:
        print(f"  could not read {p}: {exc}")
        return None

base, spec = load(sys.argv[1]), load(sys.argv[2])
if not base or not spec:
    print("  incomplete results")
    raise SystemExit(0)

print(f"  {'metric':<24} {'K=0':>10} {'K=4':>10} {'change':>10}")
for key, label in [("mean_decode_tok_s", "decode tok/s"),
                   ("mean_inclusive_tok_s", "inclusive tok/s"),
                   ("ttft_ms", "TTFT ms")]:
    b, s = base.get(key), spec.get(key)
    if b is None or s is None:
        continue
    ratio = (b / s) if key == "ttft_ms" else (s / b)
    print(f"  {label:<24} {b:>10.2f} {s:>10.2f} {ratio:>9.3f}x")

d, b = spec.get("mean_decode_tok_s"), base.get("mean_decode_tok_s")
if d and b:
    print()
    if d > b:
        print(f"  SPECULATION IS FASTER: {d/b:.3f}x decode throughput")
    else:
        print(f"  SPECULATION IS SLOWER: {d/b:.3f}x -- drafting costs more than it saves")
PY

echo
echo "=== verdict ==="
FAILED=0
[ "${IDENT_RC}" -ne 0 ] && {
    echo "FAIL: output differs between K=0 and K=${K}" >&2
    FAILED=1
}
[ "${ACC_SEEN}" -ne 1 ] && {
    echo "FAIL: engine never reported accept length" >&2
    FAILED=1
}
for f in bench_k0.json "bench_k${K}.json" identity.json; do
    [ -s "${RESULTS}/${f}" ] || {
        echo "FAIL: missing ${f}" >&2
        FAILED=1
    }
done
if [ "${FAILED}" -ne 0 ]; then
    echo "SPEC_VERIFY_FAIL" >&2
    exit 9
fi
echo "SPEC_VERIFY_COMPLETE"
