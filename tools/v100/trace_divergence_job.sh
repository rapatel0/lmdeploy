#!/usr/bin/env bash
# One-question job: WHERE does the K=4 stream first diverge from K=0?
#
# TM_SPEC_TRACE logs every forward's consumed input ids and every step's
# produced token, per row. The K=0 arm is ground truth; the K=4 arm's
# trace shows the first forward whose input differs, which names the
# exact step and mechanism of the leak -- measurement, not hypothesis.
# DEBUGGING-TOOLS.md: reach for the tool that measures before the tool
# that guesses.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-spectrace-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() { rc=$?; echo "$rc" >"${RESULTS}/exit_code"; [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"; echo "artifacts in ${RESULTS} (exit ${rc})"; }
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

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"

# ONE prompt. Multi-row batches interleave rows in the trace; a single
# sequence makes the two traces line-by-line comparable.
cat > /tmp/one_prompt.py <<'PYEOF'
import json, sys
from lmdeploy import GenerationConfig, TurbomindEngineConfig
from lmdeploy.api import pipeline
k = int(sys.argv[1])
pipe = pipeline("/models/Qwen3.8-27B-FP8",
    backend_config=TurbomindEngineConfig(tp=4, session_len=4096,
        num_draft_tokens=k, enable_prefix_caching=False, cache_generation="none"),
    log_level="INFO")
cfg = GenerationConfig(max_new_tokens=48, temperature=0.0, top_k=1, do_sample=False)
out = pipe(["List the first eight prime numbers, separated by commas."], gen_config=cfg)[0]
json.dump({"token_ids": list(out.token_ids or []), "text": out.text or ""},
          open(f"/tmp/trace_k{k}.json", "w"))
pipe.close()
PYEOF

echo "=== K=0 trace ==="
TM_SPEC_TRACE=1 python3 /tmp/one_prompt.py 0 2>&1 | grep -aE "\[trace\]|\[reject\]" > "${RESULTS}/trace_k0.log"
wc -l "${RESULTS}/trace_k0.log"

echo "=== K=4 trace ==="
TM_SPEC_TRACE=1 python3 /tmp/one_prompt.py 4 2>&1 | grep -aE "\[trace\]|\[reject\]" > "${RESULTS}/trace_k4.log"
wc -l "${RESULTS}/trace_k4.log"

cp /tmp/trace_k0.json /tmp/trace_k4.json "${RESULTS}/" 2>/dev/null

echo "=== first divergence ==="
python3 - "${RESULTS}" <<'PYEOF'
import json, sys, re
d = sys.argv[1]
k0 = json.load(open(f"{d}/trace_k0.json"))["token_ids"]
k4 = json.load(open(f"{d}/trace_k4.json"))["token_ids"]
n = min(len(k0), len(k4))
div = next((i for i in range(n) if k0[i] != k4[i]), None)
print(f"K=0 tokens ({len(k0)}): {k0[:16]}")
print(f"K=4 tokens ({len(k4)}): {k4[:16]}")
if div is None:
    print(f"IDENTICAL for {n} tokens" + ("" if len(k0)==len(k4) else " but lengths differ"))
else:
    print(f"FIRST DIVERGENCE at output position {div}: base={k0[div]} spec={k4[div]}")
# TP=4 prints every line 4x; dedupe consecutive duplicates for readability.
def dedupe(path):
    seen, out = None, []
    for line in open(path):
        m = re.search(r"\[trace\].*", line)
        if m and m.group(0) != seen:
            out.append(m.group(0)); seen = m.group(0)
    return out
print("--- K=4 trace around the divergence (deduped) ---")
for line in dedupe(f"{d}/trace_k4.log")[:60]:
    print(" ", line)
PYEOF

touch "${RESULTS}/completed"
echo "TRACE_COMPLETE"
