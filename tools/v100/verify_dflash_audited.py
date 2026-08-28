"""Compare K=0 and DFlash2 on the exact audited 1,000-token prompt."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bench_decode import build_sglang_prompt


def run(
    model: str,
    draft_model: str,
    tp: int,
    drafts: int,
    output_tokens: int,
    corpus: Path,
    input_tokens: int,
    expected_hash: str,
):
    from lmdeploy import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.api import pipeline

    config = TurbomindEngineConfig(
        model_format="fp8",
        tp=tp,
        max_batch_size=1,
        cache_max_entry_count=0.05,
        enable_prefix_caching=False,
        async_=0,
        num_draft_tokens=drafts,
        speculative_algorithm="dflash2",
        speculative_draft_model=draft_model,
        speculative_dflash_block_size=8,
        speculative_draft_window=2048,
    )
    pipe = pipeline(model, backend_config=config, log_level="INFO")
    try:
        tokenizer = pipe.async_engine.tokenizer
        _, prompt_ids, prompt_hash = build_sglang_prompt(tokenizer, corpus, input_tokens)
        print(f"K={drafts} prompt_sha256={prompt_hash}", flush=True)
        if prompt_hash != expected_hash:
            raise RuntimeError(f"expected prompt hash {expected_hash}, got {prompt_hash}")
        generation = GenerationConfig(
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            ignore_eos=True,
            temperature=0.0,
            top_k=1,
            do_sample=False,
        )

        async def run_ids():
            engine = pipe.async_engine
            session = engine.session_mgr.get()
            request = await engine.preprocess(
                messages=None,
                session_id=session,
                input_ids=prompt_ids,
                gen_config=generation,
            )
            result = None
            async for item in engine.generate(request, stream_response=True):
                response = item.to_response(0)
                result = response if result is None else result.extend(response)
            if result is None:
                raise RuntimeError("exact-id request returned no output")
            return result

        start = time.perf_counter()
        response = pipe._run(coro=run_ids()).result()
        elapsed = time.perf_counter() - start
        tokens = list(response.token_ids)
        print(f"K={drafts} elapsed={elapsed:.6f} tokens={len(tokens)} tok_s={len(tokens) / elapsed:.3f}")
        print(f"K={drafts} token_ids={tokens}")
        return tokens
    finally:
        pipe.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()

    baseline = run(
        args.model,
        args.draft_model,
        args.tp,
        0,
        args.output_tokens,
        Path(args.corpus),
        args.input_tokens,
        args.expected_prompt_sha256,
    )
    draft = run(
        args.model,
        args.draft_model,
        args.tp,
        7,
        args.output_tokens,
        Path(args.corpus),
        args.input_tokens,
        args.expected_prompt_sha256,
    )
    if draft != baseline:
        first = next((i for i, (a, b) in enumerate(zip(baseline, draft)) if a != b), min(len(baseline), len(draft)))
        print(f"DFLASH_AUDITED_IDENTITY_FAIL first_difference={first}")
        print(f"baseline[{first}:{first + 8}]={baseline[first : first + 8]}")
        print(f"dflash[{first}:{first + 8}]={draft[first : first + 8]}")
        return 2
    print("DFLASH_AUDITED_IDENTITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
