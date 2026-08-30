#!/usr/bin/env python3
"""Summarize a matched current-default K=0/K=7 Nsight result."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

COMMIT_RE = re.compile(r"final commit length (?P<commit>[0-9.]+), raw (?P<raw>[0-9.]+) over (?P<steps>\d+)")


def rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path}: {error}") from error


def benchmark(root: Path, k: int) -> dict[str, float | int]:
    try:
        data = json.loads((root / f"bench_k{k}.json").read_text())
        decode = float(data["mean_decode_tok_s"])
        log = (root / f"bench_k{k}.log").read_text(errors="replace")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid K={k} benchmark: {error}") from error
    result: dict[str, float | int] = {"decode_tok_s": decode}
    if k:
        matches = list(COMMIT_RE.finditer(log))
        if not matches:
            raise RuntimeError("K=7 profile has no acceptance summary")
        try:
            commit_length = float(matches[-1].group("commit"))
            result.update(
                {
                    "commit_length": commit_length,
                    "raw_length": float(matches[-1].group("raw")),
                    "verification_steps": int(matches[-1].group("steps")),
                    "normalized_cycle_ms": commit_length / decode * 1000.0,
                }
            )
        except (TypeError, ValueError, IndexError) as error:
            raise RuntimeError(f"invalid K=7 acceptance summary: {error}") from error
    return result


def nvtx(root: Path) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for row in rows(root / "k7_stats_nvtx_sum.csv"):
        name = row.get("Range", "").lstrip(":")
        if not name:
            continue
        try:
            result[name] = {
                "instances": int(row["Instances"]),
                "total_ms": int(row["Total Time (ns)"]) / 1e6,
                "average_ms": float(row["Avg (ns)"]) / 1e6,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid NVTX row {row}: {error}") from error
    return result


def kernel_class(name: str) -> str:
    if "Operand_B_Pack<__nv_fp8_e4m3>" in name and "MMA_Map<(int)8," in name:
        return "target_fp8_m8"
    if "Operand_B<__half>" in name and "MMA_Map<(int)8, (int)64" in name:
        return "target_fp16_flat_gdn"
    if "MMA_884_GROUPED" in name and "(int)256" in name:
        return "target_grouped_attention"
    if "ChunkedGdrKernel<(int)128, (int)16, (int)32" in name:
        return "target_gdn_v32"
    if name.startswith("nccl"):
        return "nccl"
    if "DFlashTopK16" in name or "DFlashGreedySelector" in name or "DFlashMergeTopK16" in name:
        return "draft_selector"
    return "other"


def kernels(root: Path, verify_instances: int) -> dict[str, dict[str, float | int]]:
    totals: dict[str, int] = {}
    launches: dict[str, int] = {}
    for row in rows(root / "k7_stats_cuda_gpu_kern_sum.csv"):
        group = kernel_class(row.get("Name", ""))
        try:
            totals[group] = totals.get(group, 0) + int(row["Total Time (ns)"])
            launches[group] = launches.get(group, 0) + int(row["Instances"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid kernel row {row}: {error}") from error
    return {
        group: {
            "total_ms": total / 1e6,
            "ms_per_target_verify_range": total / 1e6 / verify_instances,
            "launches": launches[group],
        }
        for group, total in totals.items()
    }


def cuda_api(root: Path, verify_instances: int) -> dict[str, dict[str, float | int]]:
    selected = {"cudaMallocFromPoolAsync_v11020", "cudaFreeAsync_v11020", "cudaMemcpyAsync", "cudaLaunchKernel"}
    result: dict[str, dict[str, float | int]] = {}
    for row in rows(root / "k7_stats_cuda_api_sum.csv"):
        name = row.get("Name", "")
        if name not in selected:
            continue
        try:
            calls = int(row["Num Calls"])
            result[name] = {
                "calls": calls,
                "calls_per_target_verify_range": calls / verify_instances,
                "total_ms": int(row["Total Time (ns)"]) / 1e6,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid CUDA API row {row}: {error}") from error
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    bench = {"k0": benchmark(root, 0), "k7": benchmark(root, 7)}
    ranges = nvtx(root)
    try:
        verify_instances = int(ranges["targetVerify"]["instances"])
        profiled_speedup = float(bench["k7"]["decode_tok_s"]) / float(bench["k0"]["decode_tok_s"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"cannot normalize the current profile: {error}") from error
    summary = {
        "benchmark": bench,
        "profiled_speedup": profiled_speedup,
        "nvtx": ranges,
        "kernels": kernels(root, verify_instances),
        "cuda_api": cuda_api(root, verify_instances),
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("DFLASH_CURRENT_NSYS_ANALYSIS " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
