#!/usr/bin/env python3
"""Analyze row-major, transposed-cuBLAS, and native SM70 FP16 vocabulary-head arms."""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(sys.argv[1])
ARMS = ("baseline", "transposed", "native")
COMMIT_RE = re.compile(r"final commit length ([0-9.]+), raw [0-9.]+ over (\d+) verification steps")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"DFLASH_FP16_FLAT_HEAD_ANALYSIS_FAIL: {message}")


def cycle(name: str) -> dict[str, float]:
    try:
        data = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
        matches = COMMIT_RE.findall((ROOT / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
        trials = data["trials"]
        decode = float(data["mean_decode_tok_s"])
        commits = [float(row[0]) for row in matches]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{name}: invalid benchmark artifacts: {exc}")
    if not trials or any(row.get("degenerate") for row in trials) or not commits:
        fail(f"{name}: missing or degenerate benchmark")
    commit = statistics.median(commits)
    return {"decode_tok_s": decode, "commit_length": commit, "cycle_ms": 1000.0 * commit / decode}


def kernel_metric(arm: str) -> dict[str, object]:
    try:
        db = sqlite3.connect(ROOT / f"profile_{arm}.sqlite")
        ranges = db.execute(
            """SELECT k.deviceId,COUNT(DISTINCT n.rowid) FROM NVTX_EVENTS n
               JOIN StringIds nt ON nt.id=n.textId JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
               ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
               AND k.globalPid=(r.globalTid & -16777216)
               WHERE nt.value='targetVerify' GROUP BY k.deviceId"""
        ).fetchall()
        head_ranges = db.execute(
            """SELECT k.deviceId,COUNT(DISTINCT n.rowid) FROM NVTX_EVENTS n
               JOIN StringIds nt ON nt.id=n.textId JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
               ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
               AND k.globalPid=(r.globalTid & -16777216)
               WHERE nt.value='postDecodeEmbedding' GROUP BY k.deviceId"""
        ).fetchall()
        rows = db.execute(
            """SELECT s.value,k.deviceId,k.gridX,k.gridY,k.gridZ,k.blockX,COUNT(*),SUM(k.end-k.start)
               FROM NVTX_EVENTS n JOIN StringIds nt ON nt.id=n.textId
               JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
               AND k.globalPid=(r.globalTid & -16777216) JOIN StringIds s ON s.id=k.demangledName
               WHERE nt.value='postDecodeEmbedding' GROUP BY s.value,k.deviceId,k.gridX,k.gridY,k.gridZ,k.blockX"""
        ).fetchall()
        db.close()
        expected_verify = {int(device): int(count) for device, count in ranges}
        expected_head = {int(device): int(count) for device, count in head_ranges}
        converted = [
            (str(name), int(device), int(gx), int(gy), int(gz), int(bx), int(count), int(total))
            for name, device, gx, gy, gz, bx, count, total in rows
        ]
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        fail(f"profile_{arm}: invalid profile: {exc}")
    selected = []
    for row in converted:
        name, _device, gx, gy, _gz, _bx, _count, _total = row
        lower = name.lower()
        native = "gemm_kernel" in lower and "operand_b<" in lower and "half" in lower and "operand_b_pack" not in lower
        cublas = "cutlass" in lower and "gemm" in lower
        if native or cublas:
            selected.append(row)
    got: dict[int, int] = {}
    for row in selected:
        got[row[1]] = got.get(row[1], 0) + row[6]
    native_by_device: dict[int, int] = {}
    for row in selected:
        if (
            "gemm_kernel" in row[0].lower()
            and "operand_b<" in row[0].lower()
            and "half" in row[0].lower()
            and "operand_b_pack" not in row[0].lower()
        ):
            native_by_device[row[1]] = native_by_device.get(row[1], 0) + row[6]
    cublas_by_device = {device: got.get(device, 0) - native_by_device.get(device, 0) for device in got}
    if sorted(expected_verify) != [0, 1, 2, 3] or sorted(expected_head) != [0, 1, 2, 3]:
        fail(f"{arm}: missing four-rank range coverage verify={expected_verify} head={expected_head}")
    route_complete = True
    if arm == "baseline":
        route_complete = cublas_by_device == expected_verify
    elif arm == "transposed":
        route_complete = cublas_by_device == expected_head
    else:
        expected_native = {device: expected_head[device] - expected_verify[device] for device in expected_head}
        route_complete = cublas_by_device == expected_verify and native_by_device == expected_native
    total_ns = sum(row[7] for row in converted)
    logical = sum(expected_head.values())
    native_launches = sum(
        row[6]
        for row in selected
        if "gemm_kernel" in row[0].lower()
        and "operand_b<" in row[0].lower()
        and "half" in row[0].lower()
        and "operand_b_pack" not in row[0].lower()
    )
    return {
        "ms_per_head": total_ns / 1e6 / logical,
        "head_ranges_by_device": expected_head,
        "verify_ranges_by_device": expected_verify,
        "selected_launches_by_device": got,
        "cublas_launches_by_device": cublas_by_device,
        "native_launches_by_device": native_by_device,
        "native_launches": native_launches,
        "route_complete": route_complete,
        "signatures": sorted({f"{r[0]} grid={r[2]}x{r[3]}x{r[4]} block={r[5]}" for r in selected}),
    }


def pct(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def main() -> None:
    unprofiled = {arm: cycle(arm) for arm in ARMS}
    profiled = {arm: cycle(f"profile_{arm}") for arm in ARMS}
    kernels = {arm: kernel_metric(arm) for arm in ARMS}
    try:
        transposed_kernel_pct = pct(
            float(kernels["transposed"]["ms_per_head"]), float(kernels["baseline"]["ms_per_head"])
        )
        native_kernel_pct = pct(float(kernels["native"]["ms_per_head"]), float(kernels["baseline"]["ms_per_head"]))
        transposed_unprofiled_pct = pct(unprofiled["transposed"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"])
        unprofiled_pct = pct(unprofiled["native"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"])
        transposed_profiled_pct = pct(profiled["transposed"]["cycle_ms"], profiled["baseline"]["cycle_ms"])
        profiled_pct = pct(profiled["native"]["cycle_ms"], profiled["baseline"]["cycle_ms"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        fail(f"invalid normalized metrics: {exc}")
    winner = (
        "transposed"
        if bool(kernels["baseline"]["route_complete"])
        and bool(kernels["transposed"]["route_complete"])
        and transposed_kernel_pct <= -10.0
        and transposed_unprofiled_pct <= -1.0
        and transposed_profiled_pct <= 0.5
        else None
    )
    result = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "transposed_kernel_pct_vs_baseline": transposed_kernel_pct,
        "native_kernel_pct_vs_baseline": native_kernel_pct,
        "transposed_unprofiled_cycle_pct": transposed_unprofiled_pct,
        "unprofiled_cycle_pct": unprofiled_pct,
        "profiled_cycle_pct": profiled_pct,
        "transposed_profiled_cycle_pct": transposed_profiled_pct,
        "winner": winner,
    }
    try:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        fail(f"cannot write analysis: {exc}")
    print("DFLASH_FP16_FLAT_HEAD_ANALYSIS", json.dumps(result, sort_keys=True))
    print("DFLASH_FP16_FLAT_HEAD_WINNER", winner or "none")


if __name__ == "__main__":
    main()
