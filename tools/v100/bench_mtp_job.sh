#!/usr/bin/env bash
# Benchmark tp=4 with MTP drafting at depth 4 against the same build with
# drafting off.
#
# The comparison is the point. A single drafted number is uninterpretable:
# tok/s depends on the model, the shapes, the cache fraction and the machine,
# so "62 tok/s" says nothing about whether speculation helped. Both arms
# therefore run in the SAME job, on the same GPUs, from the same wheel, minutes
# apart -- the only difference is num_draft_tokens.
#
# Both arms also require --require-mtp. The weights must load in both cases;
# only the draft depth differs. Loading MTP in one arm and not the other would
# compare two different models and attribute the difference to speculation.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-benchmtp-${SRC_COMMIT}
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
echo "=== gpus ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "=== build ==="
cd /src
bash /src/tools/v100/build_v100.sh || {
    echo "FAIL: build" >&2
    exit 2
}

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || {
    echo "FAIL: no wheel" >&2
    exit 2
}
echo "wheel: ${WHEEL}"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

# Inherited from the island runner, which sets the campaign defaults. Reading
# them here rather than repeating literals means the deployment config is the
# single place these values are decided.
MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
TP="${TP:-4}"
K="${NUM_DRAFT_TOKENS:-4}"
IN_TOK="${IN_TOK:-1024}"
OUT_TOK="${OUT_TOK:-256}"
TRIALS="${TRIALS:-3}"
cd /

echo
echo "=== configuration ==="
echo "  MODEL_DIR         = ${MODEL}"
echo "  TP                = ${TP}"
echo "  NUM_DRAFT_TOKENS  = ${K}"
echo "  input/output tok  = ${IN_TOK}/${OUT_TOK}, trials=${TRIALS}"
if [ "${TP}" != "4" ]; then
    echo "FAIL: this island is four V100s; tp=${TP} is not a valid configuration" >&2
    exit 3
fi

run_arm() {
    # $1 = label, $2 = num_draft_tokens, $3 = json path
    echo
    echo "=== bench: tp=4, num_draft_tokens=$2 ($1) ==="
    stdbuf -oL -eL python3 /src/tools/v100/bench_decode.py \
        --model "${MODEL}" \
        --tp "${TP}" \
        --num-draft-tokens "$2" \
        --input-tokens "${IN_TOK}" \
        --output-tokens "${OUT_TOK}" \
        --trials "${TRIALS}" \
        --require-mtp \
        --json-out "$3" 2>&1
    return "${PIPESTATUS[0]}"
}

run_arm baseline 0 "${RESULTS}/bench_k0.json"
BASE_RC=$?
run_arm "drafting depth ${K}" "${K}" "${RESULTS}/bench_k4.json"
SPEC_RC=$?

echo
echo "=== did drafting actually run in the K=${K} arm? ==="
# An engine-side acceptance line is the only proof the draft path executed.
# num_draft_tokens=4 reaching the config is necessary but not sufficient: the
# guard can still cap max_extend to 0 and skip every draft, which would produce
# a "baseline" number wearing a speculation label.
ACC="$(grep -aoE '\[MTP\] step-1 draft acceptance: [0-9]+/[0-9]+ = [0-9.]+%' "${RESULTS}/console.log" | tail -3)"
if [ -z "${ACC}" ]; then
    echo "FAIL: no acceptance line; the draft path never ran, so the K=4 arm" >&2
    echo "measures the baseline under a different name." >&2
    DRAFT_RAN=0
else
    echo "${ACC}"
    DRAFT_RAN=1
fi

echo
echo "=== comparison ==="
python3 - "${RESULTS}/bench_k0.json" "${RESULTS}/bench_k4.json" <<'PY'
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
    print("  incomplete results; no comparison")
    raise SystemExit(0)

print(f"  {'metric':<28} {'K=0':>12} {'K=4':>12} {'change':>12}")
for key, label, unit in [
    ("mean_decode_tok_s", "decode tok/s", ""),
    ("mean_inclusive_tok_s", "inclusive tok/s", ""),
    ("ttft_ms", "TTFT ms", ""),
    ("prefill_tok_s", "prefill tok/s", ""),
]:
    b, s = base.get(key), spec.get(key)
    if b is None or s is None:
        print(f"  {label:<28} {'n/a':>12} {'n/a':>12} {'':>12}")
        continue
    # TTFT is a cost: lower is better. Everything else is a rate.
    ratio = (b / s) if key == "ttft_ms" else (s / b)
    print(f"  {label:<28} {b:>12.2f} {s:>12.2f} {ratio:>11.3f}x")

print()
print(f"  mtp_enabled: K=0 {base.get('mtp_enabled')}, K=4 {spec.get('mtp_enabled')}")
print(f"  num_draft_tokens recorded: K=0 {base.get('num_draft_tokens')}, "
      f"K=4 {spec.get('num_draft_tokens')}")

# State the ceiling next to the measurement. Drafts are discarded today, so a
# speedup above 1.0 would be suspicious rather than good news, and saying so
# here stops a stray number being read as a win later.
d = spec.get("mean_decode_tok_s")
b = base.get("mean_decode_tok_s")
if d and b:
    print()
    print("  Drafts are computed and discarded in this build, so the expected")
    print("  result is K=4 <= K=0: extra work, no accepted tokens. A ratio")
    print("  above 1.0 would indicate a measurement fault, not a speedup.")
PY

echo
echo "=== verdict ==="
FAILED=0
[ "${BASE_RC}" -ne 0 ] && {
    echo "FAIL: baseline arm rc=${BASE_RC}" >&2
    FAILED=1
}
[ "${SPEC_RC}" -ne 0 ] && {
    echo "FAIL: K=4 arm rc=${SPEC_RC}" >&2
    FAILED=1
}
[ "${DRAFT_RAN}" -ne 1 ] && {
    echo "FAIL: draft path did not run in the K=4 arm" >&2
    FAILED=1
}
for f in bench_k0.json bench_k4.json; do
    [ -s "${RESULTS}/${f}" ] || {
        echo "FAIL: missing or empty ${f}" >&2
        FAILED=1
    }
done
if [ "${FAILED}" -ne 0 ]; then
    echo "BENCH_FAIL" >&2
    exit 9
fi
echo "BENCH_COMPLETE"
