#!/usr/bin/env python3
"""Analyze SM70 GDN value-column CTA decomposition arms."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import NoReturn

ARMS = ("v128", "v64", "v32", "v16")
VALUE_COLS = {"v128": 128, "v64": 64, "v32": 32, "v16": 16}
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw [0-9.]+ over (?P<steps>\d+) verification steps")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_GDN_VALUE_COLS_ANALYSIS_FAIL: {message}")


def cycle(root: Path, name: str) -> dict[str, float]:
    try:
        data = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        trials = data["trials"]
        decode = float(data["mean_decode_tok_s"])
        matches = list(COMMIT_RE.finditer((root / f"{name}.log").read_text(encoding="utf-8", errors="replace")))
        commits = [float(match.group("commit")) for match in matches]
        steps = [int(match.group("steps")) for match in matches]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{name}: invalid benchmark artifacts: {exc}")
    if not trials or any(row.get("degenerate") for row in trials) or not commits:
        fail(f"{name}: missing or degenerate trials")
    commit = statistics.median(commits)
    step_count = statistics.median(steps)
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "verification_steps": step_count,
        "cycle_ms": commit / decode * 1000.0,
    }


def gdn_kernel(path: Path, value_cols: int) -> dict[str, float | int]:
    marker = f"ChunkedGdrKernel<(int)128, (int)16, (int){value_cols},"
    total_ns = 0
    launches = 0
    try:
        with path.open(newline="", encoding="utf-8") as src:
            for row in csv.reader(src):
                joined = ",".join(row)
                if marker in joined and "__half, __half" in joined:
                    total_ns += int(row[1])
                    launches += int(row[2])
    except (OSError, IndexError, ValueError) as exc:
        fail(f"{path}: invalid kernel summary: {exc}")
    if total_ns <= 0 or launches <= 0 or launches % 48 != 0:
        fail(f"{path}: missing or invalid V{value_cols} GDN kernel totals")
    logical_ranges = launches / 48.0
    return {
        "total_ms": total_ns / 1e6,
        "launches": launches,
        "logical_ranges": logical_ranges,
        "ms_per_48_layer_range": total_ns / 1e6 / logical_ranges,
        "us_per_launch": total_ns / 1e3 / launches,
    }


def pct(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir

    unprofiled = {arm: cycle(root, arm) for arm in ARMS}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in ARMS}
    kernels = {arm: gdn_kernel(root / f"profile_{arm}_stats_cuda_gpu_kern_sum.csv", VALUE_COLS[arm]) for arm in ARMS}
    baseline_launches = kernels["v128"]["launches"]
    if any(row["launches"] != baseline_launches for row in kernels.values()):
        fail(f"GDN launch-count mismatch: {kernels}")

    try:
        gdn_ms = {arm: float(kernels[arm]["ms_per_48_layer_range"]) for arm in ARMS}
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid normalized GDN kernel metrics: {exc}")

    changes: dict[str, dict[str, float]] = {}
    for arm in ARMS[1:]:
        changes[arm] = {
            "unprofiled_cycle_pct": pct(unprofiled[arm]["cycle_ms"], unprofiled["v128"]["cycle_ms"]),
            "profiled_cycle_pct": pct(profiled[arm]["cycle_ms"], profiled["v128"]["cycle_ms"]),
            "gdn_kernel_pct": pct(gdn_ms[arm], gdn_ms["v128"]),
        }
    eligible = [
        arm
        for arm, change in changes.items()
        if change["unprofiled_cycle_pct"] <= -1.0
        and change["profiled_cycle_pct"] <= 0.5
        and change["gdn_kernel_pct"] <= -10.0
    ]
    winner = min(eligible, key=lambda arm: changes[arm]["unprofiled_cycle_pct"]) if eligible else None
    result = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "changes": changes,
        "winner": winner,
    }
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for arm, change in changes.items():
        print(
            "DFLASH_GDN_VALUE_COLS_ANALYSIS "
            f"arm={arm} unprofiled_cycle_pct={change['unprofiled_cycle_pct']:.3f} "
            f"profiled_cycle_pct={change['profiled_cycle_pct']:.3f} "
            f"gdn_kernel_pct={change['gdn_kernel_pct']:.3f}"
        )
    print(f"DFLASH_GDN_VALUE_COLS_WINNER {winner or 'none'}")


if __name__ == "__main__":
    main()
