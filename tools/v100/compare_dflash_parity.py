#!/usr/bin/env python3
"""Compare ordered first-block boundaries from LMDeploy and SGLang traces."""

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


def load_rank(root: Path, prefix: str, rank: int = 0):
    matches = sorted(root.glob(f"{prefix}{rank}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one rank-{rank} directory below {root}, got {matches}")
    directory = matches[0]
    records = {}
    for line_number, line in enumerate((directory / "manifest.jsonl").read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid manifest JSON at {directory}:{line_number}") from exc
        records[record["name"]] = record
    return directory, records


def load_tensor(directory: Path, record: dict) -> np.ndarray:
    dtype = record["dtype"]
    data = np.fromfile(directory / record["file"], dtype=DTYPES[dtype])
    if dtype == "bf16":
        data = (data.astype(np.uint32) << 16).view(np.float32)
    return data.reshape(record["shape"])


def compare_numeric(left: np.ndarray, right: np.ndarray) -> dict:
    left = left.astype(np.float64, copy=False).reshape(-1)
    right = right.astype(np.float64, copy=False).reshape(-1)
    delta = left - right
    denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    try:
        different = int(np.count_nonzero(delta))
        max_abs = float(np.max(np.abs(delta), initial=0.0))
        rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
        max_rel = float(np.max(np.abs(delta) / denom, initial=0.0))
    except (FloatingPointError, TypeError, ValueError) as exc:
        raise RuntimeError("failed to reduce numeric parity statistics") from exc
    return {
        "equal": bool(np.array_equal(left, right)),
        "allclose_f16": bool(np.allclose(left, right, rtol=2e-3, atol=2e-3)),
        "different": different,
        "max_abs": max_abs,
        "rms": rms,
        "max_rel": max_rel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lm_dir, lm = load_rank(args.lmdeploy, "rank-")
    sg_dir, sg = load_rank(args.sglang, "rank-")
    ordered = [
        ("target.post_layer_residual", "target.post_layer_residual", "numeric"),
        ("context.fc", "context.fc", "numeric"),
        ("context.norm", "context.norm", "numeric"),
        ("block.ids", "block.ids", "ids"),
        ("block.embedding", "block.embedding", "numeric"),
        ("block.initial_norm", "layer0.attention.norm_output", "numeric"),
    ]
    for layer in range(5):
        prefix = f"layer{layer}"
        ordered.extend(
            [
                (f"{prefix}.attention.conv_delta", f"{prefix}.attention.conv_delta", "right_half"),
                (f"{prefix}.attention.conv_side0", f"{prefix}.attention.conv_side0", "numeric"),
                (f"{prefix}.attention.conv_side1", f"{prefix}.attention.conv_side1", "numeric"),
                # SGLang applies the post-attention residual norm inside this
                # layer; TurboMind reports the same boundary as attention.norm.
                (f"{prefix}.attention.norm_output", f"{prefix}.mlp.norm_output", "numeric"),
                (f"{prefix}.mlp.conv_delta", f"{prefix}.mlp.conv_delta", "right_half"),
                (f"{prefix}.mlp.conv_side0", f"{prefix}.mlp.conv_side0", "numeric"),
                (f"{prefix}.mlp.conv_side1", f"{prefix}.mlp.conv_side1", "numeric"),
                # TurboMind eagerly applies the next residual norm at the end
                # of this layer. SGLang defers it to the next layer entrance.
                (
                    f"{prefix}.mlp.norm_output",
                    f"layer{layer + 1}.attention.norm_output" if layer < 4 else "block.final_norm",
                    "numeric",
                ),
            ]
        )
    ordered.extend(
        [
            ("selector.candidate_ids", "selector.candidate_ids", "ids"),
            ("selector.unary_scores", "selector.unary_scores", "numeric"),
            ("selector.scores", "selector.scores", "numeric"),
            ("selector.selected_ids", "selector.selected_ids", "ids"),
        ]
    )
    report = []
    earliest = None
    for lm_name, sg_name, kind in ordered:
        if lm_name not in lm or sg_name not in sg:
            row = {"lmdeploy": lm_name, "sglang": sg_name, "status": "missing"}
        else:
            left = load_tensor(lm_dir, lm[lm_name])
            right = load_tensor(sg_dir, sg[sg_name])
            if (
                kind == "right_half"
                and left.ndim == right.ndim
                and left.shape[:-1] == right.shape[:-1]
                and left.shape[-1] == 2 * right.shape[-1]
            ):
                left = left[..., right.shape[-1] :]
                kind = "numeric"
            if left.size != right.size:
                row = {
                    "lmdeploy": lm_name,
                    "sglang": sg_name,
                    "status": "shape_mismatch",
                    "lm_shape": list(left.shape),
                    "sg_shape": list(right.shape),
                }
            elif kind == "ids":
                equal = bool(np.array_equal(left.reshape(-1), right.reshape(-1)))
                try:
                    different = int(np.count_nonzero(left.reshape(-1) != right.reshape(-1)))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"failed to compare ID tensors {lm_name} and {sg_name}") from exc
                left_flat = left.reshape(-1)
                right_flat = right.reshape(-1)
                mismatch_indices = np.flatnonzero(left_flat != right_flat)[:32]
                row = {
                    "lmdeploy": lm_name,
                    "sglang": sg_name,
                    "status": "match" if equal else "mismatch",
                    "different": different,
                    "mismatch_indices": mismatch_indices.tolist(),
                    "lm_values": left_flat[mismatch_indices].tolist(),
                    "sg_values": right_flat[mismatch_indices].tolist(),
                }
            else:
                stats = compare_numeric(left, right)
                row = {
                    "lmdeploy": lm_name,
                    "sglang": sg_name,
                    "status": "match" if stats["allclose_f16"] else "mismatch",
                    **stats,
                }
        report.append(row)
        if earliest is None and row["status"] != "match":
            earliest = row
    result = {"earliest_mismatch": earliest, "comparisons": report}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if earliest is None:
        print("DFLASH_CROSS_RUNTIME_PREFIX_PARITY_PASS")
    else:
        print(
            "DFLASH_CROSS_RUNTIME_EARLIEST_MISMATCH "
            f"lmdeploy={earliest['lmdeploy']} sglang={earliest['sglang']} status={earliest['status']}"
        )


if __name__ == "__main__":
    main()
