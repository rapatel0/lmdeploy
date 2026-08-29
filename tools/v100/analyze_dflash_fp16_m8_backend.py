#!/usr/bin/env python3
"""Analyze attributed FP16 M<=8 cuBLAS-to-native backend substitution arms."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import NoReturn

ARMS = ("baseline", "native_gdn", "native_head", "native_both")
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw [0-9.]+ over (?P<steps>\d+) verification steps")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP16_M8_BACKEND_ANALYSIS_FAIL: {message}")


def cycle(root: Path, arm: str) -> dict[str, float]:
    try:
        data = json.loads((root / f"{arm}.json").read_text(encoding="utf-8"))
        trials = data["trials"]
        decode = float(data["mean_decode_tok_s"])
        matches = list(COMMIT_RE.finditer((root / f"{arm}.log").read_text(encoding="utf-8", errors="replace")))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{arm}: invalid benchmark artifact: {exc}")
    if not trials or any(row.get("degenerate") for row in trials) or not matches:
        fail(f"{arm}: incomplete benchmark")
    try:
        commit = statistics.median(float(match.group("commit")) for match in matches)
        steps = statistics.median(int(match.group("steps")) for match in matches)
    except (TypeError, ValueError, statistics.StatisticsError) as exc:
        fail(f"{arm}: invalid commit records: {exc}")
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "steps": steps,
        "cycle_ms": commit / decode * 1000.0,
    }


def kernel_totals(path: Path) -> dict[str, float | int]:
    totals: dict[str, float | int] = {
        "cutlass_ms": 0.0,
        "cutlass_launches": 0,
        "native_m8_ms": 0.0,
        "native_m8_launches": 0,
    }
    try:
        with path.open(newline="", encoding="utf-8") as src:
            for row in csv.reader(src):
                joined = ",".join(row)
                try:
                    duration_ms = int(row[1]) / 1e6
                    launches = int(row[2])
                except (IndexError, ValueError):
                    continue
                if "cutlass_70_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2" in joined:
                    totals["cutlass_ms"] = float(totals["cutlass_ms"]) + duration_ms
                    totals["cutlass_launches"] = int(totals["cutlass_launches"]) + launches
                if "gemm_kernel" in joined and "MMA_Map<(int)8," in joined and "Operand_B<__half>" in joined:
                    totals["native_m8_ms"] = float(totals["native_m8_ms"]) + duration_ms
                    totals["native_m8_launches"] = int(totals["native_m8_launches"]) + launches
    except OSError as exc:
        fail(f"{path}: {exc}")
    return totals


def pct(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    valid = [arm for arm in ARMS if (root / f"{arm}.json").is_file() and (root / f"profile_{arm}.json").is_file()]
    if tuple(valid) != ARMS:
        fail(f"complete four-arm matrix required, found {valid}")

    unprofiled = {arm: cycle(root, arm) for arm in valid}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in valid}
    kernels = {arm: kernel_totals(root / f"profile_{arm}_stats_cuda_gpu_kern_sum.csv") for arm in valid}
    baseline_kernels = kernels["baseline"]
    if baseline_kernels["cutlass_launches"] <= 0 or baseline_kernels["cutlass_ms"] <= 0:
        fail("baseline profile does not contain the attributed CUTLASS kernel")
    substitution = {}
    for arm in ARMS[1:]:
        substitution[arm] = (
            kernels[arm]["cutlass_launches"] < baseline_kernels["cutlass_launches"]
            and kernels[arm]["native_m8_launches"] > baseline_kernels["native_m8_launches"]
            and kernels[arm]["native_m8_ms"] > 0
        )
        if not substitution[arm]:
            fail(f"{arm}: Nsight does not prove CUTLASS-to-native substitution: {kernels[arm]}")
    changes = {
        arm: {
            "unprofiled_cycle_pct": pct(unprofiled[arm]["cycle_ms"], unprofiled["baseline"]["cycle_ms"]),
            "profiled_cycle_pct": pct(profiled[arm]["cycle_ms"], profiled["baseline"]["cycle_ms"]),
        }
        for arm in valid
        if arm != "baseline"
    }
    eligible = [
        arm
        for arm, change in changes.items()
        if substitution[arm] and change["unprofiled_cycle_pct"] <= -1.0 and change["profiled_cycle_pct"] <= 0.5
    ]
    winner = min(eligible, key=lambda arm: changes[arm]["unprofiled_cycle_pct"]) if eligible else None
    result = {
        "valid_arms": valid,
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "substitution_proven": substitution,
        "changes": changes,
        "winner": winner,
    }
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for arm, change in changes.items():
        print(
            "DFLASH_FP16_M8_BACKEND_ANALYSIS "
            f"arm={arm} unprofiled_cycle_pct={change['unprofiled_cycle_pct']:.3f} "
            f"profiled_cycle_pct={change['profiled_cycle_pct']:.3f}"
        )
    print(f"DFLASH_FP16_M8_BACKEND_WINNER {winner or 'none'}")


if __name__ == "__main__":
    main()
