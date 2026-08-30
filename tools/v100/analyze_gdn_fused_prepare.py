#!/usr/bin/env python3
"""Analyze exact SM70 Q=8 GDN preparation fusion arms."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ARMS = ("legacy", "beta", "full")
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw (?P<raw>[0-9.]+) over (?P<steps>\d+)")


def cycle(root: Path, name: str) -> dict[str, float | int]:
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


def kernel_counts(path: Path) -> dict[str, int]:
    counts = {"beta": 0, "normalize": 0, "conv": 0}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = row.get("Name", "")
                calls = int(row.get("Instances", row.get("Num Calls", "0")) or "0")
                if "ComputeBetaGKernel" in name:
                    counts["beta"] += calls
                if "L2NormalizeQKKernel" in name:
                    counts["normalize"] += calls
                if "fused_conv1d_batched_kernel_v2" in name:
                    counts["conv"] += calls
    except (OSError, csv.Error, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid kernel summary {path}: {error}") from error
    return counts


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    unprofiled = {arm: cycle(root, arm) for arm in ARMS}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in ARMS}
    kernels = {arm: kernel_counts(root / f"profile_{arm}_stats_cuda_gpu_kern_sum.csv") for arm in ARMS}
    try:
        legacy = float(unprofiled["legacy"]["normalized_cycle_ms"])
        full = float(unprofiled["full"]["normalized_cycle_ms"])
        profile_legacy = float(profiled["legacy"]["normalized_cycle_ms"])
        profile_full = float(profiled["full"]["normalized_cycle_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot compare fused preparation arms: {error}") from error
    acceptance_matched = all(
        unprofiled[arm]["commit_length"] == unprofiled["legacy"]["commit_length"]
        and unprofiled[arm]["raw_length"] == unprofiled["legacy"]["raw_length"]
        for arm in ARMS
    )
    helpers_removed = (
        kernels["legacy"]["beta"] > 0
        and kernels["legacy"]["normalize"] > 0
        and kernels["beta"]["beta"] == 0
        and kernels["full"]["beta"] == 0
        and kernels["full"]["normalize"] == 0
    )
    cycle_pct = pct(full, legacy)
    profile_pct = pct(profile_full, profile_legacy)
    summary = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "comparison": {
            "full_unprofiled_cycle_pct": cycle_pct,
            "full_profiled_cycle_pct": profile_pct,
            "acceptance_matched": acceptance_matched,
            "helpers_removed": helpers_removed,
            "material_gain": acceptance_matched and helpers_removed and cycle_pct <= -1.0 and profile_pct <= 0.0,
        },
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("GDN_FUSED_PREPARE_ANALYSIS " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
