#!/usr/bin/env python3
"""Summarize the SM70 M=8 fused E4M3 decode experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import NoReturn

ARMS = ("baseline", "fast", "combined")
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw [0-9.]+ over (?P<steps>\d+) verification steps")
M8_FP8_MARKERS = ("gemm_kernel", "MMA_Map<(int)8,", "Operand_B_Pack<__nv_fp8_e4m3>")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP8_M8_FUSED_DECODE_ANALYSIS_FAIL: {message}")


def read_cycle(root: Path, name: str) -> dict[str, float]:
    try:
        data = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        decode = float(data["mean_decode_tok_s"])
        records = [
            (float(match.group("commit")), int(match.group("steps")))
            for match in COMMIT_RE.finditer((root / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{name}: invalid benchmark artifacts: {exc}")
    trials = data.get("trials", [])
    if not trials or any(row.get("degenerate") for row in trials):
        fail(f"{name}: missing or degenerate trials")
    if not records:
        fail(f"{name}: no final commit record")
    commit = statistics.median(row[0] for row in records)
    steps = statistics.median(row[1] for row in records)
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "verification_steps": steps,
        "cycle_ms": commit / decode * 1000.0,
    }


def read_m8_fp8_ms(path: Path) -> float:
    total_ns = 0
    try:
        with path.open(newline="", encoding="utf-8") as src:
            for row in csv.reader(src):
                joined = ",".join(row)
                if all(marker in joined for marker in M8_FP8_MARKERS):
                    total_ns += int(row[1])
    except (OSError, IndexError, ValueError) as exc:
        fail(f"{path}: invalid kernel summary: {exc}")
    if total_ns <= 0:
        fail(f"{path}: no M=8 block-FP8 kernel rows")
    return total_ns / 1e6


def read_nvtx_instances(path: Path) -> int:
    try:
        count = 0
        with path.open(newline="", encoding="utf-8") as src:
            for row in csv.reader(src):
                if len(row) >= 10 and row[9].lstrip(":") == "targetVerify":
                    count += int(row[2])
    except (OSError, IndexError, ValueError) as exc:
        fail(f"{path}: invalid NVTX summary: {exc}")
    if count <= 0:
        fail(f"{path}: no targetVerify ranges")
    return count


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir

    unprofiled = {arm: read_cycle(root, arm) for arm in ARMS}
    profiled = {arm: read_cycle(root, f"profile_{arm}") for arm in ARMS}
    profile_m8: dict[str, dict[str, float | int]] = {}
    for arm in ARMS:
        total = read_m8_fp8_ms(root / f"profile_{arm}_stats_cuda_gpu_kern_sum.csv")
        ranges = read_nvtx_instances(root / f"profile_{arm}_stats_nvtx_sum.csv")
        profile_m8[arm] = {
            "total_ms": total,
            "target_verify_ranges": ranges,
            "ms_per_target_verify_range": total / ranges,
        }

    changes: dict[str, dict[str, float]] = {}
    for arm in ARMS[1:]:
        candidate_m8 = profile_m8[arm]["ms_per_target_verify_range"]
        baseline_m8 = profile_m8["baseline"]["ms_per_target_verify_range"]
        if not isinstance(candidate_m8, (int, float)) or not isinstance(baseline_m8, (int, float)):
            fail("invalid normalized M=8 profile values")
        changes[arm] = {
            "unprofiled_cycle_pct": pct(unprofiled[arm]["cycle_ms"], unprofiled["baseline"]["cycle_ms"]),
            "profiled_cycle_pct": pct(profiled[arm]["cycle_ms"], profiled["baseline"]["cycle_ms"]),
            "profile_m8_fp8_pct": pct(candidate_m8, baseline_m8),
        }

    summary = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "profile_m8_fp8": profile_m8,
        "changes": changes,
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for arm in ARMS[1:]:
        change = changes[arm]
        print(
            "DFLASH_FP8_M8_FUSED_DECODE_ANALYSIS "
            f"arm={arm} unprofiled_cycle_pct={change['unprofiled_cycle_pct']:.3f} "
            f"profiled_cycle_pct={change['profiled_cycle_pct']:.3f} "
            f"profile_m8_fp8_pct={change['profile_m8_fp8_pct']:.3f}"
        )


if __name__ == "__main__":
    main()
