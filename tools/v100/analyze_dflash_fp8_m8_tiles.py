#!/usr/bin/env python3
"""Summarize SM70 FP8 M=8 tile selection and DFlash cycle attribution."""

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
    r"\[tune\] device=(?P<device>\d+) (?P<problem>\S*_8x\d+x\d+_\d+) "
    r"(?P<kernel>\S+) swizzle=\d+ splits=\d+ measured=(?P<measured>[0-9.eE+-]+)"
)
TILE_RE = re.compile(r"_(?P<m>8)x(?P<n>64|128|256)x(?P<k>32|64|128)_2_")
OCCUPANCY_RE = re.compile(
    r"register: (?P<kernel>\S+), shared: (?P<shared>\d+) KB, regs: (?P<regs>\d+), "
    r"local: (?P<local>[0-9.]+) bytes, max_active_ctas: (?P<ctas>\d+)"
)
M8_FP8_MARKERS = (
    "gemm_kernel",
    "MMA_Map<(int)8,",
    "Operand_B_Pack<__nv_fp8_e4m3>",
)
AUDITED_PROBLEMS = {
    "gate_up_proj": (8, 8704, 5120),
    "down_proj": (8, 5120, 4352),
    "linear_attn_in_proj_qkvz": (8, 4096, 5120),
    "full_attn_qkvz": (8, 3584, 5120),
    "out_or_o_proj": (8, 5120, 1536),
    "lm_head": (8, 62080, 5120),
}
EXPECTED_DEVICES = {0, 1, 2, 3}
EXPECTED_CANDIDATE_TILES = {
    "8x64x32",
    "8x64x64",
    "8x64x128",
    "8x128x32",
    "8x128x128",
    "8x256x32",
    "8x256x64",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP8_M8_TILE_ANALYSIS_FAIL: {message}")


def read_cycle(result_dir: Path, name: str) -> dict[str, float]:
    try:
        data = json.loads((result_dir / f"{name}.json").read_text(encoding="utf-8"))
        decode = float(data["mean_decode_tok_s"])
        records = [
            (float(match.group("commit")), int(match.group("steps")))
            for match in COMMIT_RE.finditer((result_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{name}: invalid benchmark artifacts: {exc}")
    trials = data.get("trials", [])
    if not trials or any(trial.get("degenerate") for trial in trials):
        fail(f"{name}: missing or degenerate trials")
    if not records:
        fail(f"{name}: no final commit record")
    commit = statistics.median(record[0] for record in records)
    steps = statistics.median(record[1] for record in records)
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
    if total_ns == 0:
        fail(f"{path}: no SM70 block-FP8 M=8 kernel rows")
    return total_ns / 1e6


def read_nvtx_instances(path: Path, range_name: str) -> int:
    instances = 0
    try:
        with path.open(newline="", encoding="utf-8") as src:
            for row in csv.reader(src):
                if len(row) >= 10 and row[9].lstrip(":") == range_name:
                    instances += int(row[2])
    except (OSError, IndexError, ValueError) as exc:
        fail(f"{path}: invalid NVTX summary: {exc}")
    if instances <= 0:
        fail(f"{path}: no {range_name} NVTX instances")
    return instances


def kernel_tile(kernel: str) -> str | None:
    match = TILE_RE.search(kernel)
    return "x".join(match.group(name) for name in ("m", "n", "k")) if match else None


def read_selected_configs(path: Path) -> list[dict[str, object]]:
    best: dict[tuple[int, str], tuple[float, str]] = {}
    measured_tiles: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        fail(f"{path}: cannot read tuner log: {exc}")
    for line in lines:
        match = TUNE_RE.search(line)
        if not match or "e4m3" not in match.group("kernel"):
            continue
        try:
            device = int(match.group("device"))
            measured = float(match.group("measured"))
        except ValueError as exc:
            fail(f"{path}: invalid tuner record: {exc}")
        problem = match.group("problem")
        kernel = match.group("kernel")
        tile = kernel_tile(kernel)
        if tile in EXPECTED_CANDIDATE_TILES:
            measured_tiles.add(tile)
        key = (device, problem)
        if key not in best or measured < best[key][0]:
            best[key] = (measured, kernel)
    devices = {key[0] for key in best}
    if devices != EXPECTED_DEVICES:
        fail(f"{path}: expected tuner records for devices {sorted(EXPECTED_DEVICES)}, got {sorted(devices)}")
    missing_tiles = EXPECTED_CANDIDATE_TILES - measured_tiles
    if missing_tiles:
        fail(f"{path}: candidate tiles were not measured: {sorted(missing_tiles)}")

    selected = []
    for device in sorted(EXPECTED_DEVICES):
        for projection, shape in AUDITED_PROBLEMS.items():
            marker = f"_{shape[0]}x{shape[1]}x{shape[2]}_"
            matches = [
                (problem, value)
                for (record_device, problem), value in best.items()
                if record_device == device and marker in problem
            ]
            if not matches:
                fail(f"{path}: no device-{device} tuner selection for {projection} shape={shape}")
            problem, value = min(matches, key=lambda item: item[1][0])
            selected.append(
                {
                    "device": device,
                    "projection": projection,
                    "problem": problem,
                    "kernel": value[1],
                    "tile": kernel_tile(value[1]),
                    "measured_ms": value[0],
                }
            )
    return selected


def read_candidate_occupancy(path: Path) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        fail(f"{path}: cannot read occupancy log: {exc}")
    for line in lines:
        match = OCCUPANCY_RE.search(line)
        if not match:
            continue
        tile = kernel_tile(match.group("kernel"))
        if tile not in EXPECTED_CANDIDATE_TILES:
            continue
        try:
            record = {
                "tile": tile,
                "shared_kib": int(match.group("shared")),
                "registers": int(match.group("regs")),
                "local_bytes": float(match.group("local")),
                "max_active_ctas": int(match.group("ctas")),
            }
        except ValueError as exc:
            fail(f"{path}: invalid occupancy record: {exc}")
        if record["registers"] <= 0 or record["max_active_ctas"] <= 0:
            fail(f"{path}: invalid occupancy record: {record}")
        records[tile] = record
    missing = EXPECTED_CANDIDATE_TILES - records.keys()
    if missing:
        fail(f"{path}: missing occupancy records for {sorted(missing)}")
    return [records[tile] for tile in sorted(records)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir

    profile_baseline = read_cycle(root, "profile_baseline")
    profile_candidates = read_cycle(root, "profile_candidates")
    baseline_m8_total = read_m8_fp8_ms(root / "profile_baseline_stats_cuda_gpu_kern_sum.csv")
    candidate_m8_total = read_m8_fp8_ms(root / "profile_candidates_stats_cuda_gpu_kern_sum.csv")
    baseline_verify_ranges = read_nvtx_instances(root / "profile_baseline_stats_nvtx_sum.csv", "targetVerify")
    candidate_verify_ranges = read_nvtx_instances(root / "profile_candidates_stats_nvtx_sum.csv", "targetVerify")
    summary = {
        "unprofiled": {
            "baseline": read_cycle(root, "baseline"),
            "candidates": read_cycle(root, "candidates"),
        },
        "profiled": {
            "baseline": profile_baseline,
            "candidates": profile_candidates,
        },
        "profile_m8_fp8": {
            "baseline": {
                "total_ms": baseline_m8_total,
                "target_verify_ranges": baseline_verify_ranges,
                "ms_per_target_verify_range": baseline_m8_total / baseline_verify_ranges,
            },
            "candidates": {
                "total_ms": candidate_m8_total,
                "target_verify_ranges": candidate_verify_ranges,
                "ms_per_target_verify_range": candidate_m8_total / candidate_verify_ranges,
            },
        },
        "selected_configs": read_selected_configs(root / "candidates.log"),
        "candidate_occupancy": read_candidate_occupancy(root / "candidates.log"),
    }

    for section in ("unprofiled", "profiled"):
        baseline = summary[section]["baseline"]["cycle_ms"]
        candidate = summary[section]["candidates"]["cycle_ms"]
        summary[section]["cycle_change_pct"] = (candidate / baseline - 1.0) * 100.0
    baseline_ms = summary["profile_m8_fp8"]["baseline"]["ms_per_target_verify_range"]
    candidate_ms = summary["profile_m8_fp8"]["candidates"]["ms_per_target_verify_range"]
    summary["profile_m8_fp8_change_pct"] = (candidate_ms / baseline_ms - 1.0) * 100.0

    output = root / "analysis.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "DFLASH_FP8_M8_TILE_ANALYSIS "
        f"unprofiled_cycle_pct={summary['unprofiled']['cycle_change_pct']:.3f} "
        f"profiled_cycle_pct={summary['profiled']['cycle_change_pct']:.3f} "
        f"profile_m8_fp8_pct={summary['profile_m8_fp8_change_pct']:.3f} "
        f"selected_shapes={len(summary['selected_configs'])}"
    )
    for selected in summary["selected_configs"]:
        print(
            "SM70_FP8_M8_SELECTED "
            f"device={selected['device']} projection={selected['projection']} problem={selected['problem']} "
            f"tile={selected['tile']} measured_ms={selected['measured_ms']}"
        )
    for occupancy in summary["candidate_occupancy"]:
        print(
            "SM70_FP8_M8_OCCUPANCY "
            f"tile={occupancy['tile']} registers={occupancy['registers']} "
            f"local_bytes={occupancy['local_bytes']} max_active_ctas={occupancy['max_active_ctas']}"
        )


if __name__ == "__main__":
    main()
