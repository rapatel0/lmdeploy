"""Prove the MTP draft path executes, and that it changes nothing yet.

Two claims, checked separately, because they fail in different ways.

1. The draft ran. The predictor emits one record naming how many tokens it
   drafted. Without that record the path is dead code: `mtp_predictor_` is
   null, or `num_draft_tokens` never reached the engine, and the run looks
   completely normal because a draft that never happens costs nothing and
   changes nothing.

2. The output is unchanged. Drafts are currently discarded, so speculation
   must be invisible in the text. Generating the same prompt with drafting off
   and on and comparing the two strings is the only way to catch a draft that
   is corrupting the target's KV or recurrent state. That corruption would not
   raise; it would quietly degrade the text.

The second check is the one that matters right now. The draft layer writes
into its own KV slot and runs attention over the live sequence, so a mistake
in the slot wiring would show up here as different output rather than as an
error.

This driver deliberately does not report a speedup. Drafts are discarded, so
speculation can only cost time at this stage. A token rate that drops is the
expected result, not a regression.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys


def read_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"FAIL: no config.json at {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL: {path} is not valid JSON: {exc}") from None
    except OSError as exc:
        raise SystemExit(f"FAIL: cannot read {path}: {exc}") from None


def text_section(cfg: dict) -> dict:
    """Return the language-model sub-config.

    This checkpoint is multimodal, so the layer fields live under text_config
    and the top level returns None for all of them.
    """
    section = cfg.get("text_config")
    if isinstance(section, dict) and "num_hidden_layers" in section:
        return section
    return cfg


def generate(model_dir: str, tp: int, num_draft: int, prompt: str, max_new_tokens: int = 64) -> tuple[str, str]:
    """Load the model with the given draft depth and return (text, log)."""
    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    lm_logger = logging.getLogger("lmdeploy")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    lm_logger.addHandler(handler)
    lm_logger.setLevel(logging.INFO)

    # TM_LOG_INFO writes to the process stderr from C++, which the Python
    # handler above never sees.
    #
    # Do NOT redirect it. The engine aborts inside C++, which kills the
    # process outright: no exception, no finally block, no chance to copy a
    # captured file back out. Redirecting stderr therefore hides exactly the
    # message that explains the crash -- twice now.
    #
    # Instead, `tee` the whole run at the job level so the abort lands in the
    # pod log, and read the C++ records back from that same log afterwards.

    try:
        pipe = pipeline(
            model_dir,
            backend_config=TurbomindEngineConfig(
                tp=tp,
                session_len=2048,
                num_draft_tokens=num_draft,
            ),
            log_level="INFO",
        )
        # Greedy, so the two runs are comparable token for token.
        out = pipe([prompt],
                   gen_config=GenerationConfig(temperature=0.0, max_new_tokens=max_new_tokens))
        text = out[0].text if out else ""
    except Exception as exc:  # noqa: BLE001 - the message is the artifact
        lm_logger.removeHandler(handler)
        raise SystemExit(f"FAIL: generation failed at num_draft_tokens={num_draft}: {exc}") from None

    lm_logger.removeHandler(handler)
    return text, buf.getvalue()


PROMPT = "List the first five prime numbers, separated by commas."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--num-draft-tokens", type=int, default=4)
    # One generation per process. The caller runs this twice and diffs the
    # two texts. Loading both models in a single process OOMs: a 27B model at
    # tp=4 fills most of each 32GB V100, and the first pipeline's weights are
    # still resident when the second loads, because nothing frees them.
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Longer runs give the deeper draft steps a usable sample. Conditional "
        "acceptance at step 4 is only scored on rows whose first three drafts were "
        "all correct, so at 64 tokens fewer than one such row is expected.",
    )
    ap.add_argument("--emit-text", metavar="PATH", help="write the generated text here and exit")
    args = ap.parse_args()

    cfg = text_section(read_config(args.model_dir))
    if not cfg.get("mtp_num_hidden_layers"):
        print("FAIL: this checkpoint declares no MTP layer", file=sys.stderr)
        return 2

    if args.emit_text:
        text, _ = generate(args.model_dir, args.tp, args.num_draft_tokens, PROMPT, args.max_new_tokens)
        with open(args.emit_text, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"  text: {text.strip()[:160]!r}", flush=True)
        return 0

    prompt = PROMPT

    print("=== 1. baseline, drafting off ===", flush=True)
    base_text, _ = generate(args.model_dir, args.tp, 0, prompt, args.max_new_tokens)
    print(f"  text: {base_text.strip()[:160]!r}", flush=True)

    print()
    print(f"=== 2. drafting on, depth {args.num_draft_tokens} ===", flush=True)
    spec_text, _ = generate(args.model_dir, args.tp, args.num_draft_tokens, prompt, args.max_new_tokens)
    print(f"  text: {spec_text.strip()[:160]!r}", flush=True)

    # The "[MTP] drafted" records come from C++ on the real stderr, which this
    # process deliberately does not capture (see generate()). The job script
    # greps the pod log for them after this driver exits.
    print()
    print("=== 3. the output is unchanged ===")
    if base_text.strip() != spec_text.strip():
        print("FAIL: drafting altered the output. Drafts are discarded, so the", file=sys.stderr)
        print("text must be identical. This means the draft is corrupting the", file=sys.stderr)
        print("target's KV or recurrent state.", file=sys.stderr)
        print(f"  without drafting: {base_text.strip()[:200]!r}", file=sys.stderr)
        print(f"  with drafting   : {spec_text.strip()[:200]!r}", file=sys.stderr)
        return 5
    print("  identical, as required while drafts are discarded")

    print()
    print("VERIFY_MTP_DRAFT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
