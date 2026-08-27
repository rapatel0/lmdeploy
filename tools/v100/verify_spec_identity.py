"""Speculative decoding must not change the output. Prove it.

Greedy speculative decoding is an exactness-preserving optimisation. The target
model scores every drafted token and rejects any that it would not itself have
produced, so the accepted sequence is exactly what the target would have
generated alone. A faster run that produces different text has not accelerated
decoding, it has replaced the computation.

So this compares text, not quality. "Both answers are correct" is a weaker
claim that would pass while the verification logic silently accepted a wrong
token whose continuation happened to stay sensible.

Each configuration runs in its own process. The engine holds global CUDA state
and the whole point is that the two runs must not influence each other; sharing
an interpreter would let a stale allocator or cached graph make them agree for
reasons unrelated to correctness.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap

# Prompts chosen to exercise different lengths and stopping behaviour rather
# than to be answerable: a divergence at token 200 matters as much as one at
# token 3, and short greedy answers hide long-run drift.
PROMPTS = [
    "What is the capital city of Japan? Answer in one word.",
    "List the first eight prime numbers, separated by commas.",
    "Write one sentence explaining why the sky appears blue.",
    "Count from 1 to 20, separated by spaces.",
    "Name three primary colours.",
]

CHILD = textwrap.dedent(
    """
    import json, sys
    from lmdeploy import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.api import pipeline

    model_dir, tp, k, max_new, out_path = sys.argv[1:6]
    tp, k, max_new = int(tp), int(k), int(max_new)
    prompts = json.load(open(sys.argv[6]))

    pipe = pipeline(
        model_dir,
        backend_config=TurbomindEngineConfig(
            tp=tp,
            session_len=4096,
            num_draft_tokens=k,
            enable_prefix_caching=False,
            cache_generation="none",
        ),
        log_level="INFO",
    )
    # Greedy and deterministic: any difference between the two runs must come
    # from the speculative path, not from sampling.
    cfg = GenerationConfig(
        max_new_tokens=max_new,
        temperature=0.0,
        top_k=1,
        do_sample=False,
    )
    outs = pipe(prompts, gen_config=cfg)
    rows = [
        {
            "text": o.text or "",
            "token_ids": list(o.token_ids or []),
            "finish_reason": getattr(o, "finish_reason", None),
        }
        for o in outs
    ]
    json.dump(rows, open(out_path, "w"))
    pipe.close()
    """
).strip()


def run_config(model_dir: str, tp: int, k: int, max_new: int, tag: str) -> list[dict]:
    """Generate with num_draft_tokens=k in a fresh process."""
    prompts_path = f"/tmp/spec_prompts_{tag}.json"
    out_path = f"/tmp/spec_out_{tag}.json"
    script = f"/tmp/spec_child_{tag}.py"

    # Staging these files is setup, not measurement. If it fails the run has not
    # started, and saying so plainly beats a traceback that looks like an engine
    # fault in the middle of a GPU job's log.
    try:
        with open(prompts_path, "w") as handle:
            json.dump(PROMPTS, handle)
        with open(script, "w") as handle:
            handle.write(CHILD)
    except OSError as exc:
        print(f"FAIL: could not stage the child process inputs: {exc}", file=sys.stderr)
        raise SystemExit(3) from None

    proc = subprocess.run(
        [sys.executable, script, model_dir, str(tp), str(k), str(max_new), out_path, prompts_path],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # Surface the engine's own [MTP] lines: accept length is reported there and
    # the job script greps the console log for it.
    for line in (proc.stdout + proc.stderr).splitlines():
        if "[MTP]" in line or "speculation" in line.lower():
            print(f"    {line.strip()}", flush=True)

    if proc.returncode != 0:
        print(f"FAIL: generation at K={k} exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(4)

    # A missing or malformed result file means the child exited 0 without
    # writing its output. Treating that as a crash is right: the alternative is
    # comparing against nothing and reporting the two runs as identical.
    try:
        with open(out_path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"FAIL: K={k} exited 0 but produced no usable output at {out_path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(4) from None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    print("  baseline: num_draft_tokens=0", flush=True)
    base = run_config(args.model_dir, args.tp, 0, args.max_new_tokens, "k0")

    print(f"  speculative: num_draft_tokens={args.num_draft_tokens}", flush=True)
    spec = run_config(args.model_dir, args.tp, args.num_draft_tokens, args.max_new_tokens, "kn")

    if len(base) != len(spec):
        print(f"FAIL: {len(base)} baseline rows vs {len(spec)} speculative", file=sys.stderr)
        return 5

    failures: list[str] = []
    for i, (b, s) in enumerate(zip(base, spec)):
        # Token ids first: text can normalise away a real divergence, and the
        # token sequence is what the model actually produced.
        if b["token_ids"] != s["token_ids"]:
            n = min(len(b["token_ids"]), len(s["token_ids"]))
            first = next(
                (j for j in range(n) if b["token_ids"][j] != s["token_ids"][j]),
                n,
            )
            failures.append(
                f"row {i}: token ids diverge at position {first} "
                f"(base len {len(b['token_ids'])}, spec len {len(s['token_ids'])})"
            )
        elif b["text"] != s["text"]:
            failures.append(f"row {i}: same tokens but different text")

    for i, (b, s) in enumerate(zip(base, spec)):
        status = "identical" if b["token_ids"] == s["token_ids"] else "DIFFERS"
        print(
            f"  row {i}: {len(b['token_ids']):>4} tok base, {len(s['token_ids']):>4} tok spec  {status}",
            flush=True,
        )

    if args.json_out:
        try:
            with open(args.json_out, "w") as handle:
                json.dump(
                    {
                        "num_draft_tokens": args.num_draft_tokens,
                        "rows": len(base),
                        "identical": not failures,
                        "failures": failures,
                        "base_token_counts": [len(r["token_ids"]) for r in base],
                        "spec_token_counts": [len(r["token_ids"]) for r in spec],
                    },
                    handle,
                    indent=2,
                )
        except OSError as exc:
            print(f"warning: could not write {args.json_out}: {exc}", file=sys.stderr)

    if failures:
        print(file=sys.stderr)
        print(f"FAIL: speculation changed the output ({len(failures)} row(s))", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "Speculative decoding must be exactness-preserving under greedy "
            "sampling, so this is a verification defect, not a tolerance issue.",
            file=sys.stderr,
        )

        # Narrow the cause before anyone starts reading kernels.
        #
        # Two very different faults produce the same symptom here. A bug in
        # verification -- rejection, commit order, indexing -- is present at
        # every K. Recurrent-state drift is not: 48 of this model's 64 layers
        # are linear attention, and their state advances over every token the
        # forward processes, including drafts that are then rejected. Re-running
        # a position fixes KV, because KV is positional; it does not fix a
        # recurrent state, because S_t = f(S_{t-1}, x_t) has no position to
        # overwrite.
        #
        # At K=1 a rejected draft still advances that state, so drift shows up
        # there too -- but the accepted-prefix length is 1 or 2 rather than 1..5,
        # so a divergence that appears ONLY at K=4 points away from drift and
        # toward something that scales with draft count, such as indexing across
        # the K+1 block.
        print(file=sys.stderr)
        print("  narrowing: re-running at K=1", file=sys.stderr)
        k1 = run_config(args.model_dir, args.tp, 1, args.max_new_tokens, "k1")
        k1_same = all(b["token_ids"] == s["token_ids"] for b, s in zip(base, k1))
        if k1_same:
            print(
                "  K=1 is identical but K=4 is not: the fault scales with draft "
                "count, so look at K+1 block indexing before recurrent state.",
                file=sys.stderr,
            )
        else:
            print(
                "  K=1 also diverges: the fault is present with a single draft, "
                "so look at rejection, commit order, or recurrent-state drift "
                "rather than anything K-dependent.",
                file=sys.stderr,
            )
        return 6

    print()
    print(f"  all {len(base)} rows byte-identical between K=0 and K={args.num_draft_tokens}")
    print("VERIFY_SPEC_IDENTITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
