#!/usr/bin/env python3
"""Compare traced target embeddings with the audited checkpoint token row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def rank_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())


def record(directory: Path, name: str) -> dict:
    try:
        records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
        return next(item for item in records if item["name"] == name)
    except (OSError, json.JSONDecodeError, KeyError, StopIteration) as error:
        raise RuntimeError(f"cannot load {name} from {directory}: {error}") from error


def traced_embedding(directory: Path) -> np.ndarray:
    item = record(directory, "target.trajectory")
    values = np.fromfile(directory / item["file"], dtype="<f2").reshape(38, 5120)
    return values[0].astype(np.float32)


def checkpoint_embedding(model: Path, token_id: int) -> tuple[Path, str, np.ndarray]:
    candidates: list[tuple[Path, str]] = []
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("embed_tokens.weight"):
                    candidates.append((shard, key))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one embed_tokens.weight, found {[(str(p), k) for p, k in candidates]}")
    shard, key = candidates[0]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        row = handle.get_slice(key)[token_id : token_id + 1].to(torch.float32).cpu().numpy().reshape(-1)
    if row.size != 5120:
        raise RuntimeError(f"unexpected checkpoint embedding shape {row.shape}")
    return shard, key, row.astype(np.float32)


def nearest_checkpoint_rows(shard: Path, key: str, values: np.ndarray) -> list[dict[str, float | int]]:
    try:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            prefix = handle.get_slice(key)[:, :16].to(torch.float32)
        query = torch.from_numpy(values[:16]).to(torch.float32)
        errors = torch.sqrt(torch.mean((prefix - query) ** 2, dim=1))
        best = torch.topk(errors, k=5, largest=False)
        return [
            {"token_id": int(token), "prefix_rms": float(error)}
            for token, error in zip(best.indices.tolist(), best.values.tolist(), strict=True)
        ]
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot search checkpoint embeddings: {error}") from error


def summary(values: np.ndarray) -> dict[str, float | int | list[float]]:
    try:
        return {
            "nonzero": int(np.count_nonzero(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
            "first8": [float(value) for value in values[:8]],
        }
    except (TypeError, ValueError, FloatingPointError) as error:
        raise RuntimeError(f"cannot summarize embedding tensor: {error}") from error


def stats(got: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    try:
        delta = got.astype(np.float64) - ref.astype(np.float64)
        return {
            "differing": int(np.count_nonzero(got != ref)),
            "max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta * delta))),
        }
    except (TypeError, ValueError, FloatingPointError) as error:
        raise RuntimeError(f"cannot compare embedding tensors: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=198)
    args = parser.parse_args()

    shard, key, checkpoint = checkpoint_embedding(args.model, args.token_id)
    lm_dirs = rank_dirs(args.lmdeploy)
    sg_dirs = rank_dirs(args.sglang)
    if len(lm_dirs) != 4 or len(sg_dirs) != 4:
        raise RuntimeError(f"expected TP4 traces, found lm={len(lm_dirs)} sg={len(sg_dirs)}")
    first_lm = traced_embedding(lm_dirs[0])
    first_sg = traced_embedding(sg_dirs[0])
    result = {
        "checkpoint_key": key,
        "lmdeploy_nearest_checkpoint_rows": nearest_checkpoint_rows(shard, key, first_lm),
        "sglang_nearest_checkpoint_rows": nearest_checkpoint_rows(shard, key, first_sg),
        "token_id": args.token_id,
        "checkpoint": summary(checkpoint),
        "ranks": [],
    }
    for rank, (lm_dir, sg_dir) in enumerate(zip(lm_dirs, sg_dirs, strict=True)):
        lm = traced_embedding(lm_dir)
        sg = traced_embedding(sg_dir)
        result["ranks"].append(
            {
                "rank": rank,
                "lmdeploy": summary(lm),
                "sglang": summary(sg),
                "lmdeploy_vs_checkpoint": stats(lm, checkpoint),
                "sglang_vs_checkpoint": stats(sg, checkpoint),
                "lmdeploy_vs_sglang": stats(lm, sg),
            }
        )
    print("DFLASH_TARGET_EMBEDDING_ANALYSIS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
