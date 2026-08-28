"""Run target-only and DFlash2 generation with one deterministic prompt."""

from __future__ import annotations

import argparse
import time


def run(model: str, draft_model: str, tp: int, drafts: int, output_tokens: int):
    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    config = TurbomindEngineConfig(
        model_format="fp8",
        tp=tp,
        max_batch_size=1,
        cache_max_entry_count=0.05,
        async_=0,
        num_draft_tokens=drafts,
        speculative_algorithm="dflash2",
        speculative_draft_model=draft_model,
        speculative_dflash_block_size=8,
        speculative_draft_window=2048,
    )
    pipe = pipeline(model, backend_config=config, log_level="INFO")
    try:
        generation = GenerationConfig(
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            ignore_eos=True,
            temperature=0.0,
            do_sample=False,
        )
        start = time.perf_counter()
        response = pipe(["Write one short sentence about sunlight."], gen_config=generation)[0]
        elapsed = time.perf_counter() - start
        tokens = list(response.token_ids)
        print(f"K={drafts} elapsed={elapsed:.6f} tokens={len(tokens)} tok_s={len(tokens) / elapsed:.3f}")
        print(f"K={drafts} token_ids={tokens}")
        print(f"K={drafts} text={response.text!r}")
        return tokens, response.text
    finally:
        pipe.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()

    baseline_ids, baseline_text = run(args.model, args.draft_model, args.tp, 0, args.output_tokens)
    draft_ids, draft_text = run(args.model, args.draft_model, args.tp, 7, args.output_tokens)
    if draft_ids != baseline_ids:
        print("DFLASH_RUNTIME_IDENTITY_FAIL")
        print(f"baseline={baseline_text!r}")
        print(f"dflash={draft_text!r}")
        return 2
    print("DFLASH_RUNTIME_IDENTITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
