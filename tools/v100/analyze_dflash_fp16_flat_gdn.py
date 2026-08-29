#!/usr/bin/env python3
"""Analyze transposed-cuBLAS and native-flat SM70 FP16 GDN arms."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path
from typing import NoReturn

ARMS = ("baseline", "transposed", "native")
COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw [0-9.]+ over (?P<steps>\d+) verification steps")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP16_FLAT_GDN_ANALYSIS_FAIL: {message}")


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
        fail(f"{name}: missing or degenerate benchmark")
    commit = statistics.median(commits)
    return {
        "decode_tok_s": decode,
        "commit_length": commit,
        "verification_steps": statistics.median(steps),
        "cycle_ms": commit / decode * 1000.0,
    }


def kernel_metric(path: Path, native: bool) -> dict[str, object]:
    try:
        db = sqlite3.connect(path)
        ranges = db.execute(
            """SELECT k.deviceId, COUNT(DISTINCT n.rowid)
               FROM NVTX_EVENTS n JOIN StringIds nt ON nt.id=n.textId
               JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                    ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k
                    ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)
               WHERE nt.value='targetVerify' GROUP BY k.deviceId"""
        ).fetchall()
        rows = db.execute(
            """SELECT s.value, k.deviceId, k.gridX, k.gridY, k.gridZ, k.blockX,
                      COUNT(*), SUM(k.end-k.start)
               FROM NVTX_EVENTS n JOIN StringIds nt ON nt.id=n.textId
               JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                    ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k
                    ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)
               JOIN StringIds s ON s.id=k.demangledName
               WHERE nt.value='targetVerify'
               GROUP BY s.value,k.deviceId,k.gridX,k.gridY,k.gridZ,k.blockX"""
        ).fetchall()
        db.close()
    except (OSError, sqlite3.Error) as exc:
        fail(f"{path}: invalid sqlite profile: {exc}")
    try:
        converted_rows = [
            (name, int(device), int(grid_x), int(grid_y), int(grid_z), int(block_x), int(launches), int(total_ns))
            for name, device, grid_x, grid_y, grid_z, block_x, launches, total_ns in rows
        ]
    except (TypeError, ValueError) as exc:
        fail(f"{path}: invalid kernel row: {exc}")
    if native:
        fallback = sum(
            launches
            for name, _device, grid_x, grid_y, _grid_z, _block_x, launches, _total_ns in converted_rows
            if "cutlass" in name.lower()
            and "gemm" in name.lower()
            and sorted((grid_x, grid_y)) == [8, 33]
        )
        if fallback:
            fail(f"{path}: native arm retained {fallback} cuBLAS GDN fallback launches")
    selected = []
    for name, device, grid_x, grid_y, grid_z, block_x, launches, total_ns in converted_rows:
        lower = name.lower()
        if native:
            match = (
                "gemm_kernel" in lower
                and "operand_b<" in lower
                and "half" in lower
                and "operand_b_pack" not in lower
                and "mma_map" in lower
            )
        else:
            # cuBLAS CUTLASS maps M=8,N=4120 to 8x33 threadblock tiles;
            # accept either grid-axis ordering but not the 8x485 vocabulary head.
            match = (
                "cutlass" in lower
                and "gemm" in lower
                and sorted((grid_x, grid_y)) == [8, 33]
            )
        if match:
            selected.append((name, device, grid_x, grid_y, grid_z, block_x, launches, total_ns))
    if not selected:
        fail(f"{path}: missing {'native' if native else 'cuBLAS'} GDN kernel")
    try:
        expected_by_device = {int(device): 48 * int(count) for device, count in ranges}
    except (TypeError, ValueError) as exc:
        fail(f"{path}: invalid targetVerify range count: {exc}")
    if sorted(expected_by_device) != [0, 1, 2, 3]:
        fail(f"{path}: targetVerify ranges did not cover exactly devices 0..3: {expected_by_device}")
    by_device: dict[int, int] = {}
    for row in selected:
        by_device[row[1]] = by_device.get(row[1], 0) + row[6]
    if sorted(by_device) != [0, 1, 2, 3]:
        fail(f"{path}: route did not cover exactly devices 0..3: {by_device}")
    if by_device != expected_by_device:
        fail(f"{path}: incomplete or duplicate GDN route: expected {expected_by_device}, got {by_device}")
    launches = sum(row[6] for row in selected)
    total_ns = sum(row[7] for row in selected)
    expected_launches = sum(expected_by_device.values())
    if launches != expected_launches:
        fail(f"{path}: expected {expected_launches} complete GDN launches, got {launches}")
    logical_ranges = sum(count // 48 for count in expected_by_device.values())
    return {
        "total_ms": total_ns / 1e6,
        "launches": launches,
        "logical_ranges": logical_ranges,
        "ms_per_48_layer_range": total_ns / 1e6 / logical_ranges,
        "variants": len(selected),
        "launches_by_device": by_device,
        "signatures": sorted(
            {f"{row[0]} grid={row[2]}x{row[3]}x{row[4]} block={row[5]}" for row in selected}
        ),
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
    kernels = {arm: kernel_metric(root / f"profile_{arm}.sqlite", arm == "native") for arm in ARMS}
    try:
        native_ms = float(kernels["native"]["ms_per_48_layer_range"])
        transposed_ms = float(kernels["transposed"]["ms_per_48_layer_range"])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid normalized kernel metrics: {exc}")
    kernel_pct = pct(native_ms, transposed_ms)
    unprofiled_cycle_pct = pct(unprofiled["native"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"])
    profiled_cycle_pct = pct(profiled["native"]["cycle_ms"], profiled["baseline"]["cycle_ms"])
    transpose_cycle_pct = pct(unprofiled["transposed"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"])
    qualified = kernel_pct <= -10.0 and unprofiled_cycle_pct <= -1.0 and profiled_cycle_pct <= 0.5
    result = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "kernel_pct_vs_transposed": kernel_pct,
        "unprofiled_cycle_pct": unprofiled_cycle_pct,
        "profiled_cycle_pct": profiled_cycle_pct,
        "transpose_cycle_pct": transpose_cycle_pct,
        "winner": "native" if qualified else None,
    }
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "DFLASH_FP16_FLAT_GDN_ANALYSIS "
        f"kernel_pct={kernel_pct:.3f} unprofiled_cycle_pct={unprofiled_cycle_pct:.3f} "
        f"profiled_cycle_pct={profiled_cycle_pct:.3f} transpose_cycle_pct={transpose_cycle_pct:.3f}"
    )
    print(f"DFLASH_FP16_FLAT_GDN_WINNER {'native' if qualified else 'none'}")


if __name__ == "__main__":
    main()
