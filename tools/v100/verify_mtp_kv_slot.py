"""Prove the MTP draft layer received its own KV cache slot.

The slot is registered inside ``UnifiedDecoder``'s constructor, which runs on
the C++ side during model load. Nothing in Python can inspect the resulting
``cache_block_offset`` directly, so this driver checks the one observable the
registration emits, and checks it precisely.

Why a compile is not enough. ``unified_decoder.cc`` compiles whether or not the
registration branch is ever taken: the branch is guarded by
``model_weight.mtp && ...->attention``, and a checkpoint without those members
simply skips it. A build that succeeds therefore proves the code is
well-formed, not that the slot exists. This driver loads the model and demands
the record.

Why the count matters more than the record. The model has 16 full-attention
layers. If the draft layer is registered correctly it becomes attention index
16, one past the target's last. An index inside 0..15 would mean the draft
layer is sharing a slot with a target layer, which is the specific corruption
this design has to avoid: both attend over the same sequence, so a shared slot
means the draft overwrites KV entries the target still needs. That failure does
not crash. It produces subtly wrong attention for the target, which is far
harder to diagnose later than a failed assertion here.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys


def read_config(model_dir: str) -> dict:
    """Return the parsed config, or exit naming the file and the cause."""
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
    """Return the sub-config that describes the language model.

    This checkpoint is ``Qwen3_5ForConditionalGeneration``, a multimodal
    config whose top level holds only vision and routing keys. Every layer
    field lives under ``text_config``. Reading the top level returns ``None``
    for all of them, which looks exactly like a checkpoint with no MTP layer,
    so resolve the section explicitly rather than defaulting to the root.
    """
    section = cfg.get("text_config")
    if isinstance(section, dict) and "num_hidden_layers" in section:
        return section
    return cfg


def count_full_attention_layers(cfg: dict) -> int:
    """Return how many layers use full attention.

    Qwen3.5 interleaves full attention with gated DeltaNet. Only the full
    attention layers hold a KV cache, so only they consume slots, and the MTP
    slot must land immediately after the last of them.
    """
    layer_types = cfg.get("layer_types") or []
    if layer_types:
        return sum(1 for t in layer_types if "full" in str(t).lower())
    # Fall back to the interleave rule when layer_types is absent. A malformed
    # config must name itself rather than raise a bare ValueError, because this
    # runs unattended and the log is the only record.
    try:
        total = int(cfg.get("num_hidden_layers", 0))
        period = int(cfg.get("full_attention_interval", 4))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"FAIL: config.json has a non-numeric layer count: {exc}") from None
    return total // period if period else total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    raw = read_config(args.model_dir)
    cfg = text_section(raw)

    print("=== 1. what the config declares ===")
    n_full = count_full_attention_layers(cfg)
    if cfg is not raw:
        print("  (reading text_config, this is a multimodal checkpoint)")
    print(f"  num_hidden_layers      : {cfg.get('num_hidden_layers')}")
    print(f"  full-attention layers  : {n_full}")
    print(f"  mtp_num_hidden_layers  : {cfg.get('mtp_num_hidden_layers')}")
    if not cfg.get("mtp_num_hidden_layers"):
        print("FAIL: this checkpoint declares no MTP layer", file=sys.stderr)
        return 2
    if n_full <= 0:
        print("FAIL: could not determine the full-attention layer count", file=sys.stderr)
        return 3

    # The draft layer is appended after the target's attention layers, so its
    # index is exactly the number of those layers.
    expected_index = n_full
    print(f"  => expect MTP at index : {expected_index}")

    print()
    print("=== 2. load the model, capturing the C++ log ===")
    from lmdeploy import TurbomindEngineConfig, pipeline

    lm_logger = logging.getLogger("lmdeploy")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    lm_logger.addHandler(handler)
    lm_logger.setLevel(logging.INFO)

    # The TM_LOG_INFO records are written by the C++ layer to the process's
    # stderr, which the Python logging handler above does not see. Capture the
    # file descriptor as well so both sources are covered.
    saved_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    captured = ""
    try:
        pipe = pipeline(
            args.model_dir,
            backend_config=TurbomindEngineConfig(tp=args.tp, session_len=2048),
            log_level="INFO",
        )
        out = pipe(["Name one colour."], gen_config=None)
        text = out[0].text if out else ""
    except Exception as exc:  # noqa: BLE001 - the message is the artifact
        os.dup2(saved_fd, 2)
        print(f"FAIL: the model did not load: {exc}", file=sys.stderr)
        return 4
    finally:
        # Restore stderr before reading, or the read blocks on our own pipe.
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        try:
            os.set_blocking(r_fd, False)
            captured = os.read(r_fd, 1 << 22).decode("utf-8", "replace")
        except OSError:
            captured = ""
        os.close(r_fd)
        lm_logger.removeHandler(handler)

    log = captured + buf.getvalue()

    print()
    print("=== 3. the registration record ===")
    records = [ln for ln in log.splitlines() if "registered a KV slot" in ln]
    for ln in records:
        print(f"  {ln.strip()}")

    if not records:
        print(
            "FAIL: no KV slot record. The registration branch never ran, so the "
            "draft layer has no cache space of its own.",
            file=sys.stderr,
        )
        return 5

    # Check the index, not merely the record's presence. A slot inside the
    # target's range is worse than no slot, because it silently corrupts the
    # target's attention instead of failing.
    m = re.search(r"index (\d+) of (\d+)", records[0])
    if not m:
        print(f"FAIL: could not parse the index from: {records[0]}", file=sys.stderr)
        return 6
    try:
        got_index, total = int(m.group(1)), int(m.group(2))
    except ValueError as exc:  # pragma: no cover - the regex admits only digits
        print(f"FAIL: unreadable index in {records[0]!r}: {exc}", file=sys.stderr)
        return 6
    print(f"  parsed: index={got_index} total={total}")

    if got_index != expected_index:
        print(
            f"FAIL: the draft layer landed at attention index {got_index}, but the "
            f"{n_full} target layers occupy 0..{n_full - 1}. It must be at "
            f"{expected_index}. A lower index means it shares a slot with a "
            f"target layer and will overwrite KV the target still needs.",
            file=sys.stderr,
        )
        return 7
    if total != expected_index + 1:
        print(
            f"FAIL: {total} attention layers registered, expected "
            f"{expected_index + 1} (the {n_full} target layers plus one draft).",
            file=sys.stderr,
        )
        return 8

    print()
    print("=== 4. the model still generates coherent text ===")
    print(f"  {text!r}")
    if not text.strip():
        print("FAIL: empty output, the extra slot disturbed the model", file=sys.stderr)
        return 9

    print()
    print("VERIFY_MTP_KV_SLOT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
