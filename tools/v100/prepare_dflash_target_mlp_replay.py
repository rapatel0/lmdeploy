#!/usr/bin/env python3
"""Extract SGLang's TP4 layer-0 MLP-normalized final prompt row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest(path: Path) -> list[dict[str, object]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read manifest {path}: {error}") from error


def rank_dirs(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in root.iterdir():
        manifest = directory / "manifest.jsonl"
        if not directory.is_dir() or not manifest.is_file():
            continue
        records = load_manifest(manifest)
        if records:
            try:
                result[int(records[0]["tp_rank"])] = directory
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid TP rank in {manifest}") from error
    if set(result) != {0, 1, 2, 3}:
        raise RuntimeError(f"expected TP4 SGLang trace, got {sorted(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance: list[dict[str, object]] = []
    for rank, directory in sorted(rank_dirs(args.sglang).items()):
        records = load_manifest(directory / "manifest.jsonl")
        matching = [record for record in records if record["name"] == "target.trajectory"]
        if len(matching) != 1:
            raise RuntimeError(f"rank {rank} missing target trajectory")
        record = matching[0]
        if record.get("position") != 999 or record.get("token_id") != 198:
            raise RuntimeError(f"rank {rank} has invalid target alignment: {record}")
        try:
            trajectory_file = str(record["file"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"rank {rank} has invalid trajectory metadata") from error
        trajectory = np.fromfile(directory / trajectory_file, dtype="<f2").reshape(38, 5120)
        row = trajectory[4].copy()  # target.layer0.mlp_norm
        path = args.output / f"rank-{rank}.bin"
        row.tofile(path)
        provenance.append(
            {
                "rank": rank,
                "source": str(directory),
                "position": record["position"],
                "token_id": record["token_id"],
                "input_ids_sha256": record.get("input_ids_sha256"),
                "bytes": path.stat().st_size,
            }
        )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_MLP_REPLAY_PREPARED " + json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
