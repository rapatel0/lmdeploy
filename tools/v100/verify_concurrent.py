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
    return text[idx + len(marker):] if idx >= 0 else text


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

    prompts = [p for p, _ in CASES]
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=192)

    # Reference: each prompt alone, so the batch result has something to match.
    print("  establishing single-row references")
    solo: list[str] = []
    for p in prompts:
        solo.append(pipe([p], gen_config=cfg)[0].text or "")

    for rep in range(args.repeats):
        print(f"  batch round {rep}: {len(prompts)} prompts in one call")
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
            # Batching must not change greedy output.
            if text != solo[i]:
                failures.append(
                    f"{tag}: batched output differs from single-row; "
                    f"solo {final_answer(solo[i]).strip()[:60]!r} vs "
                    f"batch {final_answer(text).strip()[:60]!r}"
                )

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
        print(f"    produced {len(long_text.split())} words, finish={getattr(long_out,'finish_reason',None)}")

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
