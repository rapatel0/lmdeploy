#!/usr/bin/env python3
"""Summarize the SM70 M=8 FP8 grouped-scale reuse experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import NoReturn

COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw [0-9.]+ over (?P<steps>\d+) verification steps")
TUNE_RE = re.compile(
    r"\[tune\] (?:device=\d+ )?(?P<problem>\S*_8x\d+x\d+_\d+) "
    r"(?P<kernel>\S+) swizzle=(?P<swizzle>\d+) splits=(?P<splits>\d+) measured=(?P<measured>[0-9.eE+-]+)"
)
M8_FP8_MARKERS = ("gemm_kernel", "MMA_Map<(int)8,", "Operand_B_Pack<__nv_fp8_e4m3>")
EXPECTED_SHAPES = ((8704, 5120), (5120, 4352), (5120, 1536))


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP8_M8_REUSE_SCALE_ANALYSIS_FAIL: {message}")


def read_cycle(root: Path, name: str) -> dict[str, float]:
    try:
        data = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        records = [
            (float(match.group("commit")), int(match.group("steps")))
            for match in COMMIT_RE.finditer((root / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
        ]
        decode = float(data["mean_decode_tok_s"])
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


def selected_launches(path: Path) -> list[dict[str, object]]:
    best: dict[tuple[int, int], dict[str, object]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in TUNE_RE.finditer(text):
        problem = match.group("problem")
        shape = next((shape for shape in EXPECTED_SHAPES if f"_8x{shape[0]}x{shape[1]}_" in problem), None)
        if shape is None or "e4m3" not in match.group("kernel"):
            continue
        try:
            swizzle = int(match.group("swizzle"))
            splits = int(match.group("splits"))
            measured_ms = float(match.group("measured"))
        except ValueError:
            # Four TP executors share stdout; two valid records can interleave
            # into one malformed line. Ignore that line and use intact repeats.
            continue
        record = {
            "n": shape[0],
            "k": shape[1],
            "kernel": match.group("kernel"),
            "swizzle": swizzle,
            "splits": splits,
            "measured_ms": measured_ms,
        }
        if shape not in best or record["measured_ms"] < best[shape]["measured_ms"]:
            best[shape] = record
    if len(best) != len(EXPECTED_SHAPES):
        fail(f"{path}: expected {len(EXPECTED_SHAPES)} selected shapes, found {len(best)}")
    return [best[key] for key in sorted(best)]


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir

    unprofiled = {name: read_cycle(root, name) for name in ("baseline", "reuse")}
    profiled = {name: read_cycle(root, f"profile_{name}") for name in ("baseline", "reuse")}
    profile_m8: dict[str, dict[str, float | int]] = {}
    for name in ("baseline", "reuse"):
        total = read_m8_fp8_ms(root / f"profile_{name}_stats_cuda_gpu_kern_sum.csv")
        ranges = read_nvtx_instances(root / f"profile_{name}_stats_nvtx_sum.csv")
        profile_m8[name] = {
            "total_ms": total,
            "target_verify_ranges": ranges,
            "ms_per_target_verify_range": total / ranges,
        }

    reuse_m8_ms = profile_m8["reuse"]["ms_per_target_verify_range"]
    baseline_m8_ms = profile_m8["baseline"]["ms_per_target_verify_range"]
    if not isinstance(reuse_m8_ms, (int, float)) or not isinstance(baseline_m8_ms, (int, float)):
        fail("invalid normalized M=8 profile values")

    summary = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "profile_m8_fp8": profile_m8,
        "selected_launches": {
            "baseline": selected_launches(root / "baseline.log"),
            "reuse": selected_launches(root / "reuse.log"),
        },
        "unprofiled_cycle_change_pct": pct(unprofiled["reuse"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"]),
        "profiled_cycle_change_pct": pct(profiled["reuse"]["cycle_ms"], profiled["baseline"]["cycle_ms"]),
        "profile_m8_fp8_change_pct": pct(reuse_m8_ms, baseline_m8_ms),
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "DFLASH_FP8_M8_REUSE_SCALE_ANALYSIS "
        f"unprofiled_cycle_pct={summary['unprofiled_cycle_change_pct']:.3f} "
        f"profiled_cycle_pct={summary['profiled_cycle_change_pct']:.3f} "
        f"profile_m8_fp8_pct={summary['profile_m8_fp8_change_pct']:.3f}"
    )


if __name__ == "__main__":
    main()
