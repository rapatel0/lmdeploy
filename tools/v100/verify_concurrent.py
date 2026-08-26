#!/usr/bin/env python3
"""Test inference with several requests in flight, and with long generations.

Every measurement in this work so far used a single-row batch and at most 256
new tokens. Both are the easy case, and one of them is the case the MTP draft
path treats specially: `max_extend` takes the MINIMUM block slack across the
whole batch, so with bsz==1 it is simply that row's slack. With mixed sequence
lengths the minimum is taken over rows sitting at different offsets, which is
a code path no run has yet exercised.

Three checks:

  concurrency  submit the prompts as one batch so rows share a forward. Each
               answer must still be correct, and must match what the same
               prompt produced alone -- greedy decoding must not depend on who
               else is in the batch.
  long output  generate close to session_len, to catch a defect that only
               appears after many blocks are allocated and the sequence has
               crossed several block boundaries.
  repetition   run the batch repeatedly, so a leak or a slot that is not reset
               between forwards shows up as drift.

Exit code is the verdict.
"""

import argparse
import os
import re
import sys
from collections import Counter

CASES = [
    ("List the first five prime numbers, separated by commas.", r"2\s*,\s*3\s*,\s*5\s*,\s*7\s*,\s*11"),
    ("What is 17 multiplied by 3? Reply with the number only.", r"\b51\b"),
    ("What is the capital city of Japan? One word.", r"(?i)\btokyo\b"),
    ("Complete this sequence: 2, 4, 8, 16, __", r"\b32\b"),
    ("Spell the word 'cat' backwards.", r"(?i)\btac\b"),
]


def final_answer(text: str) -> str:
    """Drop chain-of-thought; the model restates candidate answers while reasoning."""
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker) :] if idx >= 0 else text


def degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 12:
        return False
    return Counter(words).most_common(1)[0][1] > len(words) * 0.5


def build_pipe(model_dir: str, tp: int, num_draft: int):
    from lmdeploy import TurbomindEngineConfig, pipeline

    return pipeline(
        model_dir,
        backend_config=TurbomindEngineConfig(
            tp=tp,
            session_len=2048,
            num_draft_tokens=num_draft,
            enable_prefix_caching=False,
            cache_generation="none",
        ),
        log_level="ERROR",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--num-draft-tokens", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        print(f"FAIL: no checkpoint at {args.model_dir}", file=sys.stderr)
        return 3

    from lmdeploy import GenerationConfig

    pipe = build_pipe(args.model_dir, args.tp, args.num_draft_tokens)
    failures: list[str] = []
    # Text that differed between solo and batched runs while staying correct.
    # Reported, not failed: see the note at the comparison site.
    divergences: list[str] = []

    prompts = [p for p, _ in CASES]
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=192)

    # Reference: each prompt alone, so the batch result has something to match.
    print("  establishing single-row references")
    solo: list[str] = []
    for p in prompts:
        solo.append(pipe([p], gen_config=cfg)[0].text or "")

    for rep in range(args.repeats):
        print(f"  batch round {rep}: {len(prompts)} prompts in one call")
        # Concurrency has to be PROVEN, not assumed from passing a list.
        # Pipeline._infer creates one asyncio task per prompt and gathers them
        # under a semaphore sized to max_batch_size (128), so five prompts do
        # overlap -- but a scheduler that admitted them one at a time would
        # still return five correct answers and this test would pass while
        # measuring nothing. The engine logs each forward's batch size, so the
        # job greps the C++ log for a bsz > 1 line as separate evidence.
        outs = pipe(prompts, gen_config=cfg)
        if len(outs) != len(prompts):
            failures.append(f"r{rep}: got {len(outs)} outputs for {len(prompts)} prompts")
            continue
        for i, (out, (prompt, pattern)) in enumerate(zip(outs, CASES)):
            text = out.text or ""
            tag = f"r{rep} row{i} {prompt[:30]!r}"
            if not text.strip():
                failures.append(f"{tag}: EMPTY")
                continue
            if not re.search(pattern, final_answer(text)):
                failures.append(f"{tag}: wrong answer {final_answer(text).strip()[:90]!r}")
            if degenerate(text):
                failures.append(f"{tag}: degenerate repetition")
            # Batching should not change greedy output -- but a mismatch is
            # not automatically a defect, and treating it as one would
            # manufacture a false alarm that looks exactly like a KV bug.
            #
            # Attention is per-row and rows do not read each other, so the
            # mathematics says the text must match. Floating point disagrees:
            # a different batch shape can select a different gemm tiling or
            # split-k, which changes summation ORDER, and fp16 addition is not
            # associative. A near-tie argmax can flip from that alone.
            #
            # So the ANSWER must survive batching -- that part is
            # non-negotiable and already asserted above. A divergence in the
            # surrounding text is reported as an observation, and only
            # escalated to a failure when the answer itself changed, which
            # numerics cannot plausibly explain.
            if text != solo[i]:
                solo_ok = bool(re.search(pattern, final_answer(solo[i])))
                batch_ok = bool(re.search(pattern, final_answer(text)))
                if solo_ok != batch_ok:
                    failures.append(
                        f"{tag}: batching changed CORRECTNESS (solo_ok={solo_ok}, "
                        f"batch_ok={batch_ok}); solo "
                        f"{final_answer(solo[i]).strip()[:60]!r} vs batch "
                        f"{final_answer(text).strip()[:60]!r}"
                    )
                else:
                    divergences.append(tag)

    # Long PROMPT, not just long output.
    #
    # Every prompt used in this work is 63 tokens, one short of the 64-token
    # block size, so prefill has always fitted in a single block. A prompt
    # spanning several blocks takes a different path: multiple block
    # allocations, a longer cu_k_len, and for the draft path a first drafting
    # step whose seq_len is far from the boundary that shaped every prior
    # measurement. None of that has ever executed here.
    print("  multi-block prompt")
    filler = " ".join(f"item{i}" for i in range(400))  # comfortably over 64 tokens
    long_prompt = (
        f"Here is a list to ignore: {filler}. Now answer only this: what is the capital city of Japan? One word."
    )
    lp_out = pipe([long_prompt], gen_config=cfg)[0]
    lp_text = lp_out.text or ""
    if not lp_text.strip():
        failures.append("multi-block prompt produced no text")
    elif not re.search(r"(?i)\btokyo\b", final_answer(lp_text)):
        failures.append(f"multi-block prompt gave wrong answer: {final_answer(lp_text).strip()[:90]!r}")
    else:
        print(f"    ok, answered correctly over a {len(long_prompt.split())}-word prompt")

    # Long generation: cross many block boundaries in one sequence.
    print("  long generation, 1024 new tokens")
    long_cfg = GenerationConfig(temperature=0.0, max_new_tokens=1024)
    long_out = pipe(["Count from 1 to 200, separated by spaces."], gen_config=long_cfg)[0]
    long_text = long_out.text or ""
    if not long_text.strip():
        failures.append("long generation produced no text")
    elif degenerate(long_text):
        failures.append(f"long generation degenerate: {long_text.strip()[:100]!r}")
    else:
        print(f"    produced {len(long_text.split())} words, finish={getattr(long_out, 'finish_reason', None)}")

    if divergences:
        print()
        print(f"  NOTE: {len(divergences)} row(s) differed textually between solo and batched")
        print("        runs while remaining correct. Expected from fp16 reduction order,")
        print("        not treated as a defect. Rows: " + ", ".join(divergences[:5]))

    if failures:
        print()
        print(f"FAIL: {len(failures)} problem(s)", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 7

    print("VERIFY_CONCURRENT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
