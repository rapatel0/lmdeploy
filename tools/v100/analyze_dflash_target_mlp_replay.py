#!/usr/bin/env python3
"""Measure layer-0 target MLP parity after exact input replay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw (?P<raw>[0-9.]+) over (?P<steps>\d+)")


def load_manifest(path: Path) -> list[dict[str, object]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read manifest {path}: {error}") from error


def ranks(root: Path) -> dict[int, Path]:
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
                raise RuntimeError(f"invalid rank in {manifest}") from error
    if set(result) != {0, 1, 2, 3}:
        raise RuntimeError(f"expected TP4 under {root}, got {sorted(result)}")
    return result


def trajectory(directory: Path) -> np.ndarray:
    records = load_manifest(directory / "manifest.jsonl")
    matching = [record for record in records if record["name"] == "target.trajectory"]
    if len(matching) != 1:
        raise RuntimeError(f"missing target trajectory in {directory}")
    try:
        filename = str(matching[0]["file"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid target trajectory in {directory}") from error
    return np.fromfile(directory / filename, dtype="<f2").reshape(38, 5120)


def stats(got: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    try:
        delta = got.astype(np.float32) - ref.astype(np.float32)
        differing = int(np.count_nonzero(got.view(np.uint16) != ref.view(np.uint16)))
        maximum = float(np.max(np.abs(delta)))
        rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    except (TypeError, ValueError, FloatingPointError) as error:
        raise RuntimeError(f"cannot compare target MLP tensors: {error}") from error
    return {"differing": differing, "max_abs": maximum, "rms": rms}


def benchmark(root: Path, name: str) -> dict[str, float | int]:
    try:
        data = json.loads((root / f"{name}.json").read_text())
        decode = float(data["mean_decode_tok_s"])
        matches = list(COMMIT_RE.finditer((root / f"{name}.log").read_text(errors="replace")))
        if not matches:
            raise ValueError("missing acceptance summary")
        match = matches[-1]
        commit = float(match.group("commit"))
        raw = float(match.group("raw"))
        steps = int(match.group("steps"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {name} benchmark: {error}") from error
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "raw_length": raw,
        "verification_steps": steps,
        "normalized_cycle_ms": commit / decode * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--replay-trace", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_dirs = ranks(args.baseline_trace)
    replay_dirs = ranks(args.replay_trace)
    sg_dirs = ranks(args.sglang)
    rows: list[dict[str, object]] = []
    for rank in range(4):
        baseline = trajectory(base_dirs[rank])
        replay = trajectory(replay_dirs[rank])
        sglang = trajectory(sg_dirs[rank])
        input_stats = stats(replay[4], sglang[4])
        if input_stats["differing"] != 0:
            raise RuntimeError(f"rank {rank} replay MLP input is not bit-exact: {input_stats}")
        rows.append(
            {
                "rank": rank,
                "baseline_mlp_input": stats(baseline[4], sglang[4]),
                "replay_mlp_input": input_stats,
                "baseline_mlp_output": stats(baseline[5], sglang[5]),
                "replay_mlp_output": stats(replay[5], sglang[5]),
            }
        )
    result = {
        "rows": rows,
        "performance": {
            "baseline": benchmark(args.benchmark_root, "baseline"),
            "replay": benchmark(args.benchmark_root, "replay"),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_MLP_REPLAY " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
