#!/usr/bin/env python3
"""Summarize matched contiguous-target CUDA graph arms."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw (?P<raw>[0-9.]+) over (?P<steps>\d+)")


def cycle(root: Path, name: str) -> dict[str, float]:
    try:
        data = json.loads((root / f"{name}.json").read_text())
        decode = float(data["mean_decode_tok_s"])
        matches = list(COMMIT_RE.finditer((root / f"{name}.log").read_text(errors="replace")))
        if not matches:
            raise ValueError("missing acceptance summary")
        commit = float(matches[-1].group("commit"))
        raw = float(matches[-1].group("raw"))
        steps = int(matches[-1].group("steps"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot parse {name} benchmark") from error
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "raw_length": raw,
        "verification_steps": steps,
        "normalized_cycle_ms": commit / decode * 1000.0,
    }


def cuda_calls(path: Path, prefix: str) -> int:
    total = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Name", "")
            if name.startswith(prefix):
                try:
                    total += int(row.get("Num Calls", "0").replace(",", ""))
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"invalid CUDA API count in {path}") from error
    return total


def nvtx_average(path: Path, name: str) -> float | None:
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Range") == name:
                try:
                    return float(row["Avg (ns)"]) / 1e6
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(f"invalid NVTX duration in {path}") from error
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    arms = ("baseline", "target_graph")
    unprofiled = {arm: cycle(root, arm) for arm in arms}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in arms}
    api = {}
    nvtx = {}
    for arm in arms:
        api_file = root / f"profile_{arm}_stats_cuda_api_sum.csv"
        nvtx_file = root / f"profile_{arm}_stats_nvtx_sum.csv"
        api[arm] = {
            "malloc": cuda_calls(api_file, "cudaMalloc"),
            "free": cuda_calls(api_file, "cudaFree"),
            "graph_launch": cuda_calls(api_file, "cudaGraphLaunch"),
        }
        nvtx[arm] = {"target_verify_ms": nvtx_average(nvtx_file, "targetVerify")}
    base = unprofiled["baseline"]["normalized_cycle_ms"]
    graph = unprofiled["target_graph"]["normalized_cycle_ms"]
    pbase = profiled["baseline"]["normalized_cycle_ms"]
    pgraph = profiled["target_graph"]["normalized_cycle_ms"]
    report = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "cuda_api": api,
        "nvtx": nvtx,
        "comparison": {
            "unprofiled_cycle_pct": 100.0 * (graph / base - 1.0),
            "profiled_cycle_pct": 100.0 * (pgraph / pbase - 1.0),
        },
    }
    (root / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("DFLASH_CONTIGUOUS_TARGET_GRAPH_ANALYSIS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
