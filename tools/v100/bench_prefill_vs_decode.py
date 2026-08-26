#!/usr/bin/env python3
"""Measure whether a K+1 prefill-shaped forward beats K+1 decode forwards.

Speculative decoding assumes it does. That assumption is load-bearing and has
not been tested on this hardware: a verifying forward carries input_len == K+1,
which routes it through the prefill attention path and charges it full context
temp memory, so it is NOT simply "K+1 decode steps for the price of one".

The claim is testable without any verification machinery. Decode is bound by
weight traffic: every step re-reads the model. A prefill of N tokens reads the
same weights once and does N times the arithmetic. So the per-token cost of
prefill should be far below decode, and the ratio bounds the speedup that
speculation can ever deliver at a given accept length.

Reports per-token microseconds for each, and the implied ceiling.
"""

import argparse
import json
import os
import statistics
import sys
import time


def build_pipe(model_dir: str, tp: int):
    from lmdeploy import TurbomindEngineConfig, pipeline

    return pipeline(
        model_dir,
        backend_config=TurbomindEngineConfig(tp=tp, session_len=4096),
        log_level="ERROR",
    )


def time_generation(pipe, prompt: str, new_tokens: int, reps: int) -> float:
    """Median wall time in seconds for one generation."""
    from lmdeploy import GenerationConfig

    cfg = GenerationConfig(temperature=0.0, max_new_tokens=new_tokens)
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pipe([prompt], gen_config=cfg)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3, help="repetitions; the median is reported")
    ap.add_argument("--emit-json", metavar="PATH", help="write the measurements here")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        print(f"FAIL: no checkpoint at {args.model_dir}", file=sys.stderr)
        return 3

    pipe = build_pipe(args.model_dir, args.tp)

    # Isolate prefill from decode by differencing two prompt lengths at a fixed,
    # minimal generation length. The constant per-request overhead and the one
    # decode step cancel, leaving the marginal cost of the extra prompt tokens.
    short_words = 32
    long_words = 544  # 512 more than short
    short_prompt = " ".join(["token"] * short_words)
    long_prompt = " ".join(["token"] * long_words)

    t_short = time_generation(pipe, short_prompt, 1, args.reps)
    t_long = time_generation(pipe, long_prompt, 1, args.reps)
    prefill_us = (t_long - t_short) / (long_words - short_words) * 1e6

    # Decode: difference two generation lengths at a fixed prompt, so prefill
    # and per-request overhead cancel the same way.
    t_gen_small = time_generation(pipe, short_prompt, 8, args.reps)
    t_gen_big = time_generation(pipe, short_prompt, 72, args.reps)
    decode_us = (t_gen_big - t_gen_small) / (72 - 8) * 1e6

    print(f"  prefill: {prefill_us:8.1f} us/token   (marginal, {long_words - short_words} extra prompt tokens)")
    print(f"  decode : {decode_us:8.1f} us/token   (marginal, 64 extra generated tokens)")

    if prefill_us <= 0 or decode_us <= 0:
        print("FAIL: non-positive marginal cost; timing is dominated by noise", file=sys.stderr)
        return 4

    ratio = decode_us / prefill_us
    print(f"  decode/prefill per-token cost ratio: {ratio:.1f}x")
    print()

    # A verifying forward of K+1 tokens costs roughly one decode step plus K
    # extra token-slots at prefill rates. With `a` accepted drafts per step the
    # speedup is (1 + a) tokens per that cost.
    print("  implied speedup ceiling at measured accept length:")
    for k, accepted in ((1, 0.64), (4, 0.87)):
        cost = 1.0 + k / ratio  # in units of one decode step
        print(f"    K={k}, accepted={accepted:.2f}: {(1 + accepted) / cost:.2f}x")

    if args.emit_json:
        # Never let a bad output path discard a measurement that cost a model
        # load and several generations. The numbers are already on stdout and
        # the job tees that to durable storage, so a failed write is a warning,
        # not a reason to fail the run.
        try:
            with open(args.emit_json, "w", encoding="utf-8") as f:
                json.dump({"prefill_us": prefill_us, "decode_us": decode_us, "ratio": ratio}, f)
        except OSError as exc:
            print(f"  WARNING: could not write {args.emit_json}: {exc}", file=sys.stderr)

    print("BENCH_PREFILL_VS_DECODE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
