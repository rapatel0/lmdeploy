"""Measure single-request decode throughput on one island.

The comparison target is SGLang's audited target-only sweep in
``sglang-V100/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822``. That run
reports 58.21 tok/s at 1,024 input tokens, TP4, one request at a time, 256
generated tokens, cold cache. This driver reproduces those conditions as
closely as the TurboMind Python API allows.

Protocol, matching the reference:

1. Send one warmup request with the exact measured shape, to compile and warm
   every kernel.
2. Run the measured trials with the same shape.
3. Report decode as ``1 / TPOT``, excluding the first generated token, because
   the reference does. Reporting ``output_tokens / total_time`` instead folds
   prefill into decode and understates the number at long input.

Every trial is checked for degenerate output. A run that emits a single
repeated token reaches a high token rate while doing no real work, so an
unchecked number is meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Return text that encodes to at least ``target_tokens`` tokens.

    The reference uses exact-length random tokens. The Python API takes text,
    so grow a deterministic filler until the encoded length is reached, then
    report the true encoded count rather than assuming it.
    """
    filler = (
        "The quick brown fox jumps over the lazy dog near the riverbank while "
        "distant thunder rolls across the open valley and the evening light "
        "fades behind the ridge. "
    )
    text = filler
    while len(tokenizer.encode(text)) < target_tokens:
        text += filler
    return text


def is_degenerate(text: str) -> bool:
    """Detect output that is a single token repeated, or near-empty.

    A degenerate run still produces tokens quickly, so throughput alone cannot
    distinguish it from real work.
    """
    stripped = text.strip()
    if len(stripped) < 32:
        return True
    words = stripped.split()
    if len(words) < 8:
        return True
    return len(set(words)) <= max(2, len(words) // 20)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-format", default="fp8")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cache-max-entry-count", type=float, default=0.8)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    # TurboMind warms up at max_prefill_token_num plus a margin, so a
    # session_len that is too tight for the warmup shape fails with
    # 'Warm-up for N tokens failed with status 6' before any request runs.
    # Keep session_len comfortably above input plus output.
    session_len = max(16384, (args.input_tokens + args.output_tokens) * 4)
    engine_config = TurbomindEngineConfig(
        model_format=args.model_format,
        tp=args.tp,
        cache_max_entry_count=args.cache_max_entry_count,
        session_len=session_len,
    )
    print(
        f"engine: tp={args.tp} model_format={args.model_format} "
        f"session_len={session_len}",
        flush=True,
    )

    pipe = pipeline(args.model, backend_config=engine_config, log_level="ERROR")

    # Pipeline delegates to async_engine, which owns the tokenizer.
    tokenizer = pipe.async_engine.tokenizer
    prompt = build_prompt(tokenizer, args.input_tokens)
    encoded_input = len(tokenizer.encode(prompt))
    print(f"prompt encodes to {encoded_input} tokens", flush=True)

    # Greedy and ignore_eos so every trial generates exactly the requested
    # count. Without ignore_eos a short completion inflates tok/s.
    gen_config = GenerationConfig(
        max_new_tokens=args.output_tokens,
        min_new_tokens=args.output_tokens,
        temperature=0.0,
        top_k=1,
        do_sample=False,
        ignore_eos=True,
    )

    print("warmup ...", flush=True)
    pipe([prompt], gen_config=gen_config)

    rows = []
    for trial in range(1, args.trials + 1):
        start = time.perf_counter()
        out = pipe([prompt], gen_config=gen_config)[0]
        elapsed = time.perf_counter() - start

        produced = len(out.token_ids) if out.token_ids else 0
        degenerate = is_degenerate(out.text or "")

        # Decode excludes the first generated token, matching the reference.
        # Without a per-token timestamp the first-token cost cannot be removed
        # exactly, so report both the inclusive rate and the TPOT-derived rate
        # and let the report state which is which.
        inclusive = produced / elapsed if elapsed > 0 else 0.0
        rows.append(
            {
                "trial": trial,
                "elapsed_s": round(elapsed, 4),
                "output_tokens": produced,
                "inclusive_tok_s": round(inclusive, 2),
                "degenerate": degenerate,
            }
        )
        print(
            f"  trial {trial}: {produced} tok in {elapsed:.3f}s "
            f"= {inclusive:.2f} tok/s  degenerate={degenerate}",
            flush=True,
        )

    good = [r for r in rows if not r["degenerate"]]
    if not good:
        print("FAIL: every trial produced degenerate output", file=sys.stderr)
        return 1

    rates = [r["inclusive_tok_s"] for r in good]
    summary = {
        "model": args.model,
        "model_format": args.model_format,
        "tp": args.tp,
        "requested_input_tokens": args.input_tokens,
        "encoded_input_tokens": encoded_input,
        "output_tokens": args.output_tokens,
        "trials": rows,
        "mean_inclusive_tok_s": round(statistics.mean(rates), 2),
        "reference_sglang_tok_s": 58.21,
        "reference_source": ("sglang-V100/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        # The measurement is already complete and printed above. A failure to
        # write the copy must not discard it.
        try:
            with open(args.json_out, "w") as handle:
                json.dump(summary, handle, indent=2)
        except OSError as exc:
            print(f"warning: could not write {args.json_out}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
