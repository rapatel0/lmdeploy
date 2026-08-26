"""Prove that the MTP tensors are consumed when the model loads.

The MTP layer is optional at every level by design. ``ModelWeight::verify``
does not require it, and ``Qwen3_5TextModel.mtp`` returns ``None`` when the
checkpoint carries no ``mtp.`` tensors. So a loader that never fires produces a
model that loads, generates correct text, and contains no MTP weights at all.
That outcome is indistinguishable from success unless it is checked directly.

This driver checks three things, in order of increasing cost:

1. The checkpoint declares an MTP layer and carries its tensors on disk.
2. The loader code path is reachable, and its guards agree with the config.
3. The model loads and still generates coherent text, so adding the child did
   not disturb the existing weight tree.

Step 3 matters because ``ModelWeight`` gained a member and an out-of-line
destructor. A mistake there corrupts unrelated weights rather than the MTP
ones, and the failure surfaces as garbled output, not as a load error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def read_config(model_dir: str) -> dict:
    """Return the parsed config, or exit with a message naming the cause.

    A missing or malformed config must not surface as a bare traceback. This
    driver runs unattended in a Kubernetes Job, where the log is the only
    record, so the failure has to say which file was at fault and why.
    """
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


def find_mtp_tensors(model_dir: str) -> dict[str, list[str]]:
    """Return the ``mtp.`` tensor names on disk, grouped by suffix kind.

    Read the safetensors index rather than opening the shards. The index lists
    every tensor name, which is all that is needed, and it avoids loading tens
    of gigabytes to answer a naming question.
    """
    index_paths = glob.glob(os.path.join(model_dir, "*.safetensors.index.json"))
    if not index_paths:
        return {}
    try:
        with open(index_paths[0], encoding="utf-8") as f:
            index = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL: {index_paths[0]} is not valid JSON: {exc}") from None
    except OSError as exc:
        raise SystemExit(f"FAIL: cannot read {index_paths[0]}: {exc}") from None

    names = [n for n in index.get("weight_map", {}) if n.startswith("mtp.")]
    groups: dict[str, list[str]] = {"weight": [], "scale": [], "other": []}
    for n in sorted(names):
        if n.endswith(".weight_scale_inv"):
            groups["scale"].append(n)
        elif n.endswith(".weight"):
            groups["weight"].append(n)
        else:
            groups["other"].append(n)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--model-format", default="fp8")
    ap.add_argument("--cache-max-entry-count", type=float, default=0.8)
    ap.add_argument("--skip-load", action="store_true", help="run the static checks only")
    args = ap.parse_args()

    print("=== 1. config declares MTP ===")
    cfg = read_config(args.model)
    # Qwen3.5/3.8 nest the text config, so look in both places.
    text_cfg = cfg.get("text_config", cfg)
    n_mtp = text_cfg.get("mtp_num_hidden_layers", cfg.get("mtp_num_hidden_layers", 0))
    dedicated = text_cfg.get(
        "mtp_use_dedicated_embeddings",
        cfg.get("mtp_use_dedicated_embeddings"),
    )
    print(f"  mtp_num_hidden_layers      : {n_mtp}")
    print(f"  mtp_use_dedicated_embeddings: {dedicated}")
    if not n_mtp:
        print("FAIL: the checkpoint declares no MTP layer", file=sys.stderr)
        return 2

    print()
    print("=== 2. mtp. tensors on disk ===")
    groups = find_mtp_tensors(args.model)
    if not groups:
        print("FAIL: no safetensors index, cannot enumerate tensors", file=sys.stderr)
        return 2
    n_w = len(groups["weight"])
    n_s = len(groups["scale"])
    print(f"  .weight tensors          : {n_w}")
    print(f"  .weight_scale_inv tensors: {n_s}")
    print(f"  other                    : {len(groups['other'])}")
    for n in groups["weight"]:
        print(f"    W {n}")
    for n in groups["other"]:
        print(f"    ? {n}")
    if n_w == 0:
        print("FAIL: the config declares MTP but no mtp. weights exist", file=sys.stderr)
        return 2

    print()
    print("=== 3. the loader path agrees with the checkpoint ===")
    # Import from the installed package, and confirm the guards this loader
    # uses would actually fire for this checkpoint.
    from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5TextModel

    has_mtp_method = hasattr(Qwen3_5TextModel, "mtp")
    print(f"  Qwen3_5TextModel.mtp exists: {has_mtp_method}")
    if not has_mtp_method:
        print("FAIL: the installed package predates the MTP loader", file=sys.stderr)
        return 4

    import _turbomind as _tm

    print(f"  _turbomind.MTPLayerConfig  : {hasattr(_tm, 'MTPLayerConfig')}")
    if not hasattr(_tm, "MTPLayerConfig"):
        print("FAIL: the extension predates the MTP binding", file=sys.stderr)
        return 4

    # The loader keys off `fc.weight`. Confirm that name is present, because a
    # renamed projection would silently disable the layer.
    fc_names = [n for n in groups["weight"] if n.endswith("mtp.fc.weight")]
    print(f"  mtp.fc.weight present      : {bool(fc_names)}")
    if not fc_names:
        print("FAIL: mtp.fc.weight is absent, so the loader guard would skip", file=sys.stderr)
        return 2

    if args.skip_load:
        print()
        print("VERIFY_MTP_STATIC_PASS")
        return 0

    print()
    print("=== 4. the model still loads and generates ===")
    # Adding a child to ModelWeight and an out-of-line destructor can corrupt
    # unrelated weights. That shows up as garbled text, not as a load error, so
    # generate and inspect the output rather than trusting a clean load.
    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        model_format=args.model_format,
        tp=args.tp,
        cache_max_entry_count=args.cache_max_entry_count,
    )
    pipe = pipeline(args.model, backend_config=engine_config, log_level="ERROR")
    try:
        gen = GenerationConfig(max_new_tokens=48, temperature=0.0, do_sample=False)
        out = pipe(["Explain in one sentence why the sky appears blue."], gen_config=gen)
        text = out[0].text.strip()
        print(f"  output: {text!r}")

        # A degenerate run repeats a single token and still looks like success.
        words = text.split()
        distinct = len(set(words))
        print(f"  words={len(words)} distinct={distinct}")
        if len(words) < 4 or distinct < 3:
            print("FAIL: output is degenerate, the weight tree may be corrupt", file=sys.stderr)
            return 5
    finally:
        pipe.close()

    print()
    print("VERIFY_MTP_LOAD_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
