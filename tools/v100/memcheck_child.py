"""Generate once with speculation on, under compute-sanitizer.

The smallest workload that still reaches the fault. One prompt, 32 new tokens,
no baseline arm and no benchmark: the crash appears within the first few decode
steps, so everything past that is sanitizer runtime spent for nothing.

This is a separate file rather than a heredoc because LMDeploy spawns workers
and multiprocessing spawn re-imports __main__ by path. A script arriving on
stdin is named '<stdin>', which no worker can open, and that failure surfaces
as BrokenProcessPool -- an engine fault to anyone reading the log.

No diagnostics are printed here on purpose. The deliverable is the sanitizer's
own report, which names the kernel, the address, and the allocation. Adding
printf tracing alongside it is the approach this run replaces.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    from lmdeploy import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.api import pipeline

    print(
        f"MEMCHECK_CHILD: tp={args.tp} k={args.num_draft_tokens} "
        f"max_new={args.max_new_tokens}",
        flush=True,
    )

    pipe = pipeline(
        args.model_dir,
        backend_config=TurbomindEngineConfig(
            tp=args.tp,
            session_len=4096,
            num_draft_tokens=args.num_draft_tokens,
            enable_prefix_caching=False,
            cache_generation="none",
        ),
        log_level="INFO",
    )

    # Greedy, so the run is deterministic and reproduces the same path.
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=1,
        do_sample=False,
    )

    # Two prompt shapes, run separately, because the failure mode depends on
    # the shape. Observed matrix at K=4:
    #   8-token prompt    -> hang under async, clean under launch-blocking
    #   1102-token prompt -> illegal access ~200ms after the first draft
    # A short-prompt-only probe declared the configuration clean and was wrong
    # for the long-prompt path.
    short = "Count from 1 to 20, separated by spaces."
    long = "The following is a detailed technical description. " * 130

    for tag, prompt in (("short", short), ("long", long)):
        print(f"MEMCHECK_PROMPT {tag}", flush=True)
        outs = pipe([prompt], gen_config=cfg)
        for i, o in enumerate(outs):
            print(f"MEMCHECK_ROW {tag} {i}: {len(o.token_ids or [])} tokens", flush=True)

    pipe.close()
    print("MEMCHECK_CHILD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
