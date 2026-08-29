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
        expected = {int(device): int(count) for device, count in ranges}
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
        if arm == "native":
            match = (
                "gemm_kernel" in lower
                and "operand_b<" in lower
                and "half" in lower
                and "operand_b_pack" not in lower
            )
        else:
            match = "cutlass" in lower and "gemm" in lower
        if match:
            selected.append(row)
    got: dict[int, int] = {}
    for row in selected:
        got[row[1]] = got.get(row[1], 0) + row[6]
    if sorted(expected) != [0, 1, 2, 3] or got != expected:
        fail(f"{arm}: incomplete head route expected={expected} got={got}")
    total_ns = sum(row[7] for row in selected)
    logical = sum(expected.values())
    return {
        "ms_per_head": total_ns / 1e6 / logical,
        "launches_by_device": got,
        "signatures": sorted({f"{r[0]} grid={r[2]}x{r[3]}x{r[4]} block={r[5]}" for r in selected}),
    }


def pct(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def main() -> None:
    unprofiled = {arm: cycle(arm) for arm in ARMS}
    profiled = {arm: cycle(f"profile_{arm}") for arm in ARMS}
    kernels = {arm: kernel_metric(arm) for arm in ARMS}
    try:
        kernel_pct = pct(float(kernels["native"]["ms_per_head"]), float(kernels["transposed"]["ms_per_head"]))
        unprofiled_pct = pct(unprofiled["native"]["cycle_ms"], unprofiled["baseline"]["cycle_ms"])
        profiled_pct = pct(profiled["native"]["cycle_ms"], profiled["baseline"]["cycle_ms"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        fail(f"invalid normalized metrics: {exc}")
    winner = "native" if kernel_pct <= -10.0 and unprofiled_pct <= -1.0 and profiled_pct <= 0.5 else None
    result = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "kernels": kernels,
        "kernel_pct_vs_transposed": kernel_pct,
        "unprofiled_cycle_pct": unprofiled_pct,
        "profiled_cycle_pct": profiled_pct,
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
