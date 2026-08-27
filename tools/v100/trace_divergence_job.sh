#!/usr/bin/env bash
# One-question job: WHERE does the configured speculative stream first diverge from K=0?
#
# TM_SPEC_TRACE logs every forward's consumed input ids and every step's
# produced token, per row. The K=0 arm is ground truth; the speculative
# arm's trace shows the first forward whose input differs, which names the
# exact step and mechanism of the leak -- measurement, not hypothesis.
# DEBUGGING-TOOLS.md: reach for the tool that measures before the tool
# that guesses.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-spectrace-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "$rc" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
echo "=== build ==="
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    echo "FAIL: build" >&2
    grep -aE "error:|Error [0-9]+" "${RESULTS}/build.log" | head -20
    exit 2
fi
tail -2 "${RESULTS}/build.log"
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

export TRACE_MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
K="${NUM_DRAFT_TOKENS:-4}"

# The five-row identity batch, shortened to eight outputs. The remaining
# force-reject split reproduces only with multiple rows; the same prompt alone
# is byte-identical for 16 tokens.
cat >/tmp/one_prompt.py <<'PYEOF'
import json, os, sys
from lmdeploy import GenerationConfig, TurbomindEngineConfig
from lmdeploy.api import pipeline
k = int(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "force":
    os.environ["TM_MTP_FORCE_REJECT"] = "1"
pipe = pipeline(os.environ["TRACE_MODEL"],
    backend_config=TurbomindEngineConfig(tp=4, session_len=4096,
        num_draft_tokens=k, enable_prefix_caching=False, cache_generation="none"),
    log_level="INFO")
cfg = GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1, do_sample=False)
prompts = [
    "What is the capital city of Japan? Answer in one word.",
    "List the first eight prime numbers, separated by commas.",
    "Write one sentence explaining why the sky appears blue.",
    "Count from 1 to 20, separated by spaces.",
    "Name three primary colours.",
]
outs = pipe(prompts, gen_config=cfg)
json.dump([{"token_ids": list(out.token_ids or []), "text": out.text or ""} for out in outs],
          open(f"/tmp/trace_k{k}.json", "w"))
pipe.close()
PYEOF

echo "=== K=0 trace ==="
TM_SPEC_TRACE=1 python3 /tmp/one_prompt.py 0 2>&1 |
    grep -aE "\[trace\]|\[reject\]" >"${RESULTS}/trace_k0.log"
wc -l "${RESULTS}/trace_k0.log"

echo "=== K=${K} trace ==="
TM_SPEC_TRACE=1 TM_SPEC_LOGIT_PARITY=1 python3 /tmp/one_prompt.py "${K}" force 2>&1 |
    grep -aE "\[trace\]|\[reject\]|\[logit-parity\]" >"${RESULTS}/trace_k${K}.log"
wc -l "${RESULTS}/trace_k${K}.log"

cp /tmp/trace_k0.json "/tmp/trace_k${K}.json" "${RESULTS}/" 2>/dev/null

echo "=== first divergence ==="
python3 - "${RESULTS}" "${K}" <<'PYEOF'
import json, sys, re
d, depth = sys.argv[1], sys.argv[2]
base = json.load(open(f"{d}/trace_k0.json"))
spec = json.load(open(f"{d}/trace_k{depth}.json"))
for row, (b, s) in enumerate(zip(base, spec)):
    k0, ks = b["token_ids"], s["token_ids"]
    n = min(len(k0), len(ks))
    div = next((i for i in range(n) if k0[i] != ks[i]), None)
    print(f"row {row} K=0: {k0}")
    print(f"row {row} K={depth}: {ks}")
    if div is None:
        print(f"row {row} IDENTICAL for {n} tokens" + ("" if len(k0)==len(ks) else " but lengths differ"))
    else:
        print(f"row {row} FIRST DIVERGENCE at output position {div}: base={k0[div]} spec={ks[div]}")
# TP=4 prints every line 4x; dedupe consecutive duplicates for readability.
def dedupe(path):
    seen, out = None, []
    for line in open(path):
        m = re.search(r"\[trace\].*", line)
        if m and m.group(0) != seen:
            out.append(m.group(0)); seen = m.group(0)
    return out
print(f"--- K={depth} trace around the divergence (deduped) ---")
for line in dedupe(f"{d}/trace_k{depth}.log")[-100:]:
    print(" ", line)
PYEOF

touch "${RESULTS}/completed"
echo "TRACE_COMPLETE"
