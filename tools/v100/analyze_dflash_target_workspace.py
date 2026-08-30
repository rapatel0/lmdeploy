#!/usr/bin/env python3
"""Analyze stable target-verification workspaces against dynamic allocation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ARMS = ("dynamic", "workspace")
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw (?P<raw>[0-9.]+) over (?P<steps>\d+)")


def cycle(root: Path, name: str) -> dict[str, float | int]:
    try:
        data = json.loads((root / f"{name}.json").read_text())
        decode = float(data["mean_decode_tok_s"])
        matches = list(COMMIT_RE.finditer((root / f"{name}.log").read_text(errors="replace")))
        if not matches:
            raise ValueError("missing acceptance summary")
        commit = float(matches[-1].group("commit"))
        raw = float(matches[-1].group("raw"))
        steps = int(matches[-1].group("steps"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {name} benchmark: {error}") from error
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "raw_length": raw,
        "verification_steps": steps,
        "normalized_cycle_ms": commit / decode * 1000.0,
    }


def allocator_calls(path: Path) -> dict[str, int]:
    result = {"malloc": 0, "free": 0}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = row.get("Name", "")
                if name == "cudaMallocFromPoolAsync_v11020":
                    result["malloc"] += int(row["Num Calls"])
                elif name == "cudaFreeAsync_v11020":
                    result["free"] += int(row["Num Calls"])
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid allocator summary {path}: {error}") from error
    if not result["malloc"] or not result["free"]:
        raise RuntimeError(f"missing allocator calls in {path}")
    return result


def percent(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    unprofiled = {arm: cycle(root, arm) for arm in ARMS}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in ARMS}
    allocators = {
        arm: allocator_calls(root / f"profile_{arm}_stats_cuda_api_sum.csv") for arm in ARMS
    }
    try:
        dynamic_cycle = float(unprofiled["dynamic"]["normalized_cycle_ms"])
        workspace_cycle = float(unprofiled["workspace"]["normalized_cycle_ms"])
        dynamic_profile = float(profiled["dynamic"]["normalized_cycle_ms"])
        workspace_profile = float(profiled["workspace"]["normalized_cycle_ms"])
        allocator_reduction = 1.0 - allocators["workspace"]["malloc"] / allocators["dynamic"]["malloc"]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"cannot compare target workspaces: {error}") from error
    summary = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "allocator_calls": allocators,
        "comparison": {
            "unprofiled_cycle_pct": percent(workspace_cycle, dynamic_cycle),
            "profiled_cycle_pct": percent(workspace_profile, dynamic_profile),
            "allocator_reduction_pct": allocator_reduction * 100.0,
        },
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_WORKSPACE_ANALYSIS " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
