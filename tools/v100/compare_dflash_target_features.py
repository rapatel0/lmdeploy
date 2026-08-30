#!/usr/bin/env python3
"""Compare audited prompt target residual features consumed by DFlash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def manifest(path: Path) -> list[dict[str, object]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error


def rank_dirs(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in root.iterdir():
        path = directory / "manifest.jsonl"
        if not path.is_file():
            continue
        records = manifest(path)
        if records:
            try:
                result[int(records[0]["tp_rank"])] = directory
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid TP rank in {path}") from error
    if set(result) != {0, 1, 2, 3}:
        raise RuntimeError(f"expected TP4 under {root}, got {sorted(result)}")
    return result


def tensor(directory: Path) -> np.ndarray:
    records = manifest(directory / "manifest.jsonl")
    matching = [record for record in records if record["name"] == "target.prompt_features"]
    if len(matching) != 1:
        raise RuntimeError(f"missing target.prompt_features in {directory}")
    try:
        filename = str(matching[0]["file"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid target.prompt_features metadata in {directory}") from error
    try:
        value = np.fromfile(directory / filename, dtype="<f2").reshape(5, 5120)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot read target.prompt_features in {directory}: {error}") from error
    if not np.isfinite(value).all():
        raise RuntimeError(f"non-finite target.prompt_features in {directory}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lm_dirs = rank_dirs(args.lmdeploy)
    sg_dirs = rank_dirs(args.sglang)
    feature_layers = (5, 19, 33, 47, 61)
    rows: list[dict[str, float | int]] = []
    for rank in range(4):
        lm_value = tensor(lm_dirs[rank])
        sg_value = tensor(sg_dirs[rank])
        for index, layer in enumerate(feature_layers):
            try:
                delta = lm_value[index].astype(np.float32) - sg_value[index].astype(np.float32)
                differing = int(
                    np.count_nonzero(lm_value[index].view(np.uint16) != sg_value[index].view(np.uint16))
                )
                maximum = float(np.max(np.abs(delta)))
                rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
            except (TypeError, ValueError, FloatingPointError) as error:
                raise RuntimeError(f"cannot compare rank {rank} feature {index}: {error}") from error
            rows.append(
                {
                    "rank": rank,
                    "feature_index": index,
                    "target_layer": layer,
                    "differing": differing,
                    "max_abs": maximum,
                    "rms": rms,
                }
            )
    result = {"rows": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_FEATURES " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
