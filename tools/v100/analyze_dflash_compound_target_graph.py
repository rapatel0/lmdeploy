#!/usr/bin/env python3
"""Compare baseline, contiguous-target, and target-plus-draft CUDA graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_dflash_contiguous_target_graph import cuda_calls, cycle, nvtx_average


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    arms = ("baseline", "target_graph", "compound")
    unprofiled = {arm: cycle(root, arm) for arm in arms}
    profiled = {arm: cycle(root, f"profile_{arm}") for arm in arms}
    api = {}
    nvtx = {}
    for arm in arms:
        api_file = root / f"profile_{arm}_stats_cuda_api_sum.csv"
        nvtx_file = root / f"profile_{arm}_stats_nvtx_sum.csv"
        api[arm] = {
            "malloc": cuda_calls(api_file, "cudaMalloc"),
            "free": cuda_calls(api_file, "cudaFree"),
            "graph_launch": cuda_calls(api_file, "cudaGraphLaunch"),
        }
        nvtx[arm] = {
            "target_verify_ms": nvtx_average(nvtx_file, "targetVerify"),
            "draft_select_ms": nvtx_average(nvtx_file, "dflashDraftAndSelect"),
        }
    base = unprofiled["baseline"]["normalized_cycle_ms"]
    pbase = profiled["baseline"]["normalized_cycle_ms"]
    comparison = {}
    for arm in arms[1:]:
        comparison[arm] = {
            "unprofiled_cycle_pct": 100.0 * (unprofiled[arm]["normalized_cycle_ms"] / base - 1.0),
            "profiled_cycle_pct": 100.0 * (profiled[arm]["normalized_cycle_ms"] / pbase - 1.0),
        }
    report = {
        "unprofiled": unprofiled,
        "profiled": profiled,
        "cuda_api": api,
        "nvtx": nvtx,
        "comparison": comparison,
    }
    (root / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("DFLASH_COMPOUND_TARGET_GRAPH_ANALYSIS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
