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
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=0,
        help="MTP draft depth K. 0 disables drafting and gives the baseline.",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--require-mtp",
        action="store_true",
        help="fail unless the loader reports MTP speculation enabled",
    )
    args = parser.parse_args()

    # Import the factory from lmdeploy.api directly. The package re-exports it
    # as lmdeploy.pipeline, which collides with the lmdeploy/pipeline.py module
    # of the same name; a static analyzer resolves the module and reports the
    # call as 'Module is not callable'. Importing from .api is unambiguous.
    from lmdeploy import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.api import pipeline

    # TurboMind warms up at max_prefill_token_num plus a margin, so a
    # session_len that is too tight for the warmup shape fails with
    # 'Warm-up for N tokens failed with status 6' before any request runs.
    # Keep session_len comfortably above input plus output.
    session_len = max(16384, (args.input_tokens + args.output_tokens) * 4)
    # Prefix caching OFF, deliberately.
    #
    # Every trial sends the same prompt, so with the cache enabled the first
    # trial computes the prefill and the rest reuse it. That is not the
    # configuration under test, and it does not distort both arms equally: the
    # speculative arm runs a different number of forwards, so cache hits shift
    # the K=0 and K=4 timings by different amounts and the ratio stops meaning
    # what it claims to mean.
    engine_config = TurbomindEngineConfig(
        model_format=args.model_format,
        tp=args.tp,
        cache_max_entry_count=args.cache_max_entry_count,
        session_len=session_len,
        num_draft_tokens=args.num_draft_tokens,
        enable_prefix_caching=False,
    )
    print(
        f"engine: tp={args.tp} model_format={args.model_format} "
        f"session_len={session_len} num_draft_tokens={args.num_draft_tokens}",
        flush=True,
    )

    # Capture the loader's [MTP] records so the run can state what it measured.
    # A benchmark that cannot say whether the speculation weights were loaded
    # invites attributing a number to the wrong configuration.
    import logging

    mtp_records: list[str] = []

    class _CaptureMTP(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "[MTP]" in msg:
                mtp_records.append(f"{record.levelname} {msg}")

    _lm_logger = logging.getLogger("lmdeploy")
    _mtp_handler = _CaptureMTP()
    _mtp_handler.setLevel(logging.INFO)
    _lm_logger.addHandler(_mtp_handler)

    # INFO rather than ERROR. pipeline() applies this level to the shared
    # 'lmdeploy' logger, and at ERROR the [MTP] records are suppressed before
    # the handler above can see them.
    pipe = pipeline(args.model, backend_config=engine_config, log_level="INFO")

    mtp_enabled = any("speculation ENABLED" in r for r in mtp_records)
    print("=== MTP status for this run ===", flush=True)
    for r in mtp_records:
        print(f"  {r}", flush=True)
    if not mtp_records:
        print("  no [MTP] record: the loader did not run", flush=True)
    print(f"  mtp_enabled={mtp_enabled}", flush=True)
    if args.require_mtp and not mtp_enabled:
        print(
            "FAIL: --require-mtp was given but the loader did not enable MTP",
            file=sys.stderr,
        )
        pipe.close()
        return 2

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

    # Measure TTFT separately with a one-token request. The synchronous API
    # returns only when generation completes, so the first-token latency
    # cannot be recovered from the full run. A max_new_tokens=1 request is
    # prefill plus one decode step, which is the closest available proxy and
    # slightly overstates TTFT.
    ttft_config = GenerationConfig(
        max_new_tokens=1,
        temperature=0.0,
        top_k=1,
        do_sample=False,
        ignore_eos=True,
    )
    print("measuring TTFT ...", flush=True)
    ttft_samples = []
    for _ in range(args.trials):
        start = time.perf_counter()
        pipe([prompt], gen_config=ttft_config)
        ttft_samples.append(time.perf_counter() - start)
    ttft_s = statistics.median(ttft_samples)
    prefill_tok_s = encoded_input / ttft_s if ttft_s > 0 else 0.0
    print(f"  TTFT {ttft_s * 1000:.1f} ms -> prefill {prefill_tok_s:.1f} tok/s", flush=True)

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
            f"  trial {trial}: {produced} tok in {elapsed:.3f}s = {inclusive:.2f} tok/s  degenerate={degenerate}",
            flush=True,
        )

    good = [r for r in rows if not r["degenerate"]]
    if not good:
        print("FAIL: every trial produced degenerate output", file=sys.stderr)
        return 1

    rates = [r["inclusive_tok_s"] for r in good]

    # The acceptance line is emitted by the C++ engine to stderr, not through
    # the Python logger, so the [MTP] handler above never sees it. Recover it
    # from the captured records if present; the job script also greps the
    # console log, which is the authoritative copy.
    acceptance_lines = [r for r in mtp_records if "acceptance" in r]

    # Decode excluding the first generated token, which is what the reference
    # reports. Subtracting the measured TTFT from the full run removes prefill
    # and the first decode step together.
    decode_rates = []
    for row in good:
        tail_s = row["elapsed_s"] - ttft_s
        tail_tokens = row["output_tokens"] - 1
        if tail_s > 0 and tail_tokens > 0:
            decode_rates.append(tail_tokens / tail_s)

    summary = {
        "model": args.model,
        "model_format": args.model_format,
        "tp": args.tp,
        # Without this the stored JSON cannot say which configuration produced
        # the number, and a baseline result is indistinguishable from a drafted
        # one once the console log is gone.
        "num_draft_tokens": args.num_draft_tokens,
        "requested_input_tokens": args.input_tokens,
        "encoded_input_tokens": encoded_input,
        "output_tokens": args.output_tokens,
        "ttft_ms": round(ttft_s * 1000, 2),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "trials": rows,
        # Record what was loaded alongside what was measured, so a stored
        # result cannot later be attributed to the wrong configuration.
        "mtp_enabled": mtp_enabled,
        "mtp_records": mtp_records,
        # Acceptance is the only evidence that drafting did work rather than
        # merely being switched on. A speedup with 0% acceptance would mean the
        # gain came from somewhere else and the attribution is wrong.
        "mtp_acceptance": acceptance_lines,
        "mean_inclusive_tok_s": round(statistics.mean(rates), 2),
        "mean_decode_tok_s": (round(statistics.mean(decode_rates), 2) if decode_rates else None),
        "reference_sglang_decode_tok_s": 58.21,
        "reference_sglang_prefill_tok_s": 2991.9,
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
