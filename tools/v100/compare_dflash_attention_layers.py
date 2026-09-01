#!/usr/bin/env python3
"""Compare native TP4 DFlash attention boundaries across all five layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DTYPES = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bf16": np.dtype("<u2"),
}


def rank_directories(root: Path) -> list[Path]:
    result = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
    if len(result) != 4:
        raise RuntimeError(f"expected four rank directories below {root}, got {result}")
    return result


def records(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line_number, line in enumerate((root / "manifest.jsonl").read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid manifest JSON at {root}:{line_number}") from error
        result.setdefault(record["name"], record)
    return result


def load(root: Path, manifest: dict[str, dict], name: str) -> np.ndarray:
    record = manifest[name]
    dtype = record["dtype"]
    value = np.fromfile(root / record["file"], dtype=DTYPES[dtype])
    if dtype == "bf16":
        value = (value.astype(np.uint32) << 16).view(np.float32)
    return value.reshape(record["shape"])


def rope_interleaved(value: np.ndarray) -> np.ndarray:
    shape = value.shape
    if shape[-1] % 128:
        raise RuntimeError(f"RoPE width is not head-dim-128 aligned: {shape}")
    return value.reshape(-1, shape[-1] // 128, 2, 64).transpose(0, 1, 3, 2).reshape(shape)


def statistics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool | list[int]]:
    if left.size != right.size:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    delta = left64 - right64
    try:
        different = int(np.count_nonzero(delta))
        max_abs = float(np.max(np.abs(delta), initial=0.0))
        rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    except (FloatingPointError, TypeError, ValueError) as error:
        raise RuntimeError("failed to reduce attention parity statistics") from error
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(left64, right64)),
        "allclose_f16": bool(np.allclose(left64, right64, rtol=2e-3, atol=2e-3)),
        "different": different,
        "max_abs": max_abs,
        "rms": rms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lm_roots = rank_directories(args.lmdeploy)
    sg_roots = rank_directories(args.sglang)
    rows: list[dict] = []
    for rank, (lm_root, sg_root) in enumerate(zip(lm_roots, sg_roots, strict=True)):
        lm = records(lm_root)
        sg = records(sg_root)
        for layer in range(5):
            prefix = f"layer{layer}.attention"
            required_lm = {
                f"{prefix}.qkv_projection",
                f"{prefix}.qkv_pre_process",
                f"{prefix}.qkv_post_process",
                f"{prefix}.flattened_kv",
                f"{prefix}.core_output",
            }
            required_sg = {
                f"{prefix}.qkv_projection",
                f"{prefix}.q_normalized",
                f"{prefix}.k_normalized",
                f"{prefix}.tilelang.q",
                f"{prefix}.tilelang.k",
                f"{prefix}.tilelang.v",
                f"{prefix}.tilelang.output",
            }
            if not required_lm <= lm.keys() or not required_sg <= sg.keys():
                raise RuntimeError(
                    f"rank {rank} layer {layer} missing LM={sorted(required_lm - lm.keys())} "
                    f"SG={sorted(required_sg - sg.keys())}"
                )

            lm_projection = load(lm_root, lm, f"{prefix}.qkv_projection").reshape(8, 1536)
            lm_pre = load(lm_root, lm, f"{prefix}.qkv_pre_process").reshape(8, 1536)
            lm_post = load(lm_root, lm, f"{prefix}.qkv_post_process").reshape(8, 1536)
            sg_projection = load(sg_root, sg, f"{prefix}.qkv_projection").reshape(8, 1536)
            sg_q_normalized = load(sg_root, sg, f"{prefix}.q_normalized").reshape(8, 1024)
            sg_k_normalized = load(sg_root, sg, f"{prefix}.k_normalized").reshape(8, 256)
            sg_q = load(sg_root, sg, f"{prefix}.tilelang.q").reshape(8, 1024)
            sg_k = load(sg_root, sg, f"{prefix}.tilelang.k").reshape(-1, 2, 128)
            sg_v = load(sg_root, sg, f"{prefix}.tilelang.v").reshape(-1, 2, 128)
            sg_output = load(sg_root, sg, f"{prefix}.tilelang.output").reshape(8, 1024)
            key_count = sg_k.shape[0]
            lm_flat_storage = load(lm_root, lm, f"{prefix}.flattened_kv").reshape(-1)
            needed = 2 * 2 * key_count * 128
            if lm_flat_storage.size < needed:
                raise RuntimeError(
                    f"rank {rank} layer {layer} flattened KV has {lm_flat_storage.size}, needs {needed}"
                )
            lm_flat = lm_flat_storage[:needed].reshape(2, 2, key_count, 128)

            pairs = {
                "q_projection": (lm_projection[:, :1024], rope_interleaved(sg_projection[:, :1024])),
                "k_projection": (lm_projection[:, 1024:1280], rope_interleaved(sg_projection[:, 1024:1280])),
                "v_projection": (lm_projection[:, 1280:1536], sg_projection[:, 1280:1536]),
                "q_normalized": (lm_pre[:, :1024], rope_interleaved(sg_q_normalized)),
                "k_normalized": (lm_pre[:, 1024:1280], rope_interleaved(sg_k_normalized)),
                "q_rotated": (lm_post[:, :1024], sg_q),
                "proposal_k": (lm_post[:, 1024:1280], sg_k[-8:].reshape(8, 256)),
                "proposal_v": (lm_post[:, 1280:1536], sg_v[-8:].reshape(8, 256)),
                "cache_k": (lm_flat[:, 0], sg_k.transpose(1, 0, 2)),
                "cache_v": (lm_flat[:, 1], sg_v.transpose(1, 0, 2)),
                "core_output": (load(lm_root, lm, f"{prefix}.core_output"), sg_output),
            }
            for boundary, (left, right) in pairs.items():
                row = {"rank": rank, "layer": layer, "boundary": boundary, **statistics(left, right)}
                rows.append(row)
                print("DFLASH_ATTENTION_LAYER_BOUNDARY " + json.dumps(row, sort_keys=True))

    result = {"comparisons": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DFLASH_ATTENTION_LAYER_COMPARISON_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
