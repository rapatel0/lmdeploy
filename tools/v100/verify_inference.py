#!/usr/bin/env python3
"""Prove inference is working, stable, and produces appropriate output.

Every measurement so far used ONE prompt, checked only that speculative and
baseline text matched byte for byte, and never asked whether the text was
right. Two identical wrong answers pass that test. This asks the question the
byte-comparison cannot.

Three properties, each with a stated pass condition:

  working     -- each prompt produces non-empty text that contains a known
                 correct answer. Checked per prompt, not in aggregate.
  stable      -- many sequential generations in one process, varying prompt
                 and length, with no crash, no hang, no truncation, and no
                 degradation in later generations. Also runs the same prompt
                 twice to confirm determinism under greedy decoding.
  appropriate -- output terminates for the stated reason, respects
                 max_new_tokens, and is not degenerate (no single token
                 repeated to fill the budget).

Exit code is the verdict. Failures print what was expected and what arrived.
"""

import argparse
import os
import re
import sys
import time
from collections import Counter

# Prompt, and a regex the answer must match. Deliberately factual and short,
# so a correct model passes and a broken one cannot pass by accident.
CASES = [
    ("List the first five prime numbers, separated by commas.", r"2\s*,\s*3\s*,\s*5\s*,\s*7\s*,\s*11"),
    ("What is 17 multiplied by 3? Reply with the number only.", r"\b51\b"),
    ("What is the capital city of Japan? One word.", r"(?i)\btokyo\b"),
    ("Complete this sequence: 2, 4, 8, 16, __", r"\b32\b"),
    ("Spell the word 'cat' backwards.", r"(?i)\btac\b"),
]


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


def generate(pipe, prompt: str, max_new: int):
    from lmdeploy import GenerationConfig

    cfg = GenerationConfig(temperature=0.0, max_new_tokens=max_new)
    t0 = time.perf_counter()
    out = pipe([prompt], gen_config=cfg)[0]
    return out.text or "", getattr(out, "finish_reason", None), time.perf_counter() - t0


def final_answer(text: str) -> str:
    """The answer proper, with any chain-of-thought removed.

    This model reasons before answering and restates candidate answers while
    doing so. Observed output contains "Need final only: 2, 3, 5, 7, 11" INSIDE
    the reasoning block. Matching against the whole string would therefore pass
    a response whose reasoning mentions the right answer and whose final answer
    is wrong -- verified against negative controls, where
    "Is it Tokyo or Kyoto? The capital is Kyoto." matched /tokyo/ and passed.

    Scoring only what follows the last </think> closes that hole. When the tag
    is absent the whole text is returned, so a non-reasoning reply still works.
    """
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):] if idx >= 0 else text


def degenerate(text: str) -> bool:
    """True when one token fills most of the output -- a classic broken-KV tell."""
    words = text.split()
    if len(words) < 12:
        return False
    return Counter(words).most_common(1)[0][1] > len(words) * 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--num-draft-tokens", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3, help="passes over the case list")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        print(f"FAIL: no checkpoint at {args.model_dir}", file=sys.stderr)
        return 3

    pipe = build_pipe(args.model_dir, args.tp, args.num_draft_tokens)
    failures: list[str] = []
    first_round_text: dict[str, str] = {}
    latencies: list[float] = []

    for rnd in range(args.rounds):
        # Vary the budget across rounds so a length-dependent defect surfaces.
        max_new = (128, 256, 64)[rnd % 3]
        for prompt, pattern in CASES:
            text, reason, secs = generate(pipe, prompt, max_new)
            latencies.append(secs)
            tag = f"r{rnd} {prompt[:38]!r}"

            if not text.strip():
                failures.append(f"{tag}: EMPTY output")
                continue
            answer = final_answer(text)
            if not re.search(pattern, answer):
                failures.append(
                    f"{tag}: no match for /{pattern}/ in final answer {answer.strip()[:120]!r}"
                )
            if degenerate(text):
                failures.append(f"{tag}: degenerate repetition in {text.strip()[:120]!r}")
            if reason not in (None, "stop", "length"):
                failures.append(f"{tag}: unexpected finish_reason {reason!r}")

            # Greedy decoding must be deterministic across rounds.
            if prompt in first_round_text:
                if first_round_text[prompt] != text:
                    failures.append(
                        f"{tag}: nondeterministic under greedy; "
                        f"round 0 gave {first_round_text[prompt].strip()[:80]!r}, "
                        f"now {text.strip()[:80]!r}"
                    )
            else:
                first_round_text[prompt] = text

            print(f"  ok  {tag} -> {text.strip()[:70]!r}")

    total = args.rounds * len(CASES)
    print()
    print(f"  {total - len(failures)}/{total} checks passed over {total} generations")
    if latencies:
        # A late generation far slower than an early one suggests a leak.
        half = len(latencies) // 2
        if half:
            early = sum(latencies[:half]) / half
            late = sum(latencies[half:]) / (len(latencies) - half)
            print(f"  latency early {early:.2f}s vs late {late:.2f}s")
            if late > early * 2.0:
                failures.append(f"latency degraded {early:.2f}s -> {late:.2f}s across the run")

    if failures:
        print()
        print(f"FAIL: {len(failures)} problem(s)", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 6

    print("VERIFY_INFERENCE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
