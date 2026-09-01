#!/usr/bin/env python3
"""Analyze aligned DFlash selector scores from two four-rank parity traces."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def records(root: str) -> dict[str, dict]:
    try:
        with open(Path(root, "manifest.jsonl"), encoding="utf-8") as manifest:
            return {row["name"]: row for row in map(json.loads, manifest)}
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"invalid parity manifest under {root}: {exc}") from exc


def load(root: str, rec: dict, name: str) -> np.ndarray:
    row = rec[name]
    dtype = {"f16": "<f2", "f32": "<f4", "i32": "<i4"}[row["dtype"]]
    return np.fromfile(Path(root, row["file"]), dtype=dtype).reshape(row["shape"])


def as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"cannot convert {value!r} to int") from exc


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"cannot convert {value!r} to float") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdeploy", required=True)
    ap.add_argument("--sglang", required=True)
    args = ap.parse_args()
    lm_roots = sorted(glob.glob(args.lmdeploy + "/rank-*-pid-*"))
    sg_roots = sorted(glob.glob(args.sglang + "/rank-*-pid-*"))
    if len(lm_roots) != 4 or len(sg_roots) != 4:
        raise RuntimeError(f"expected four trace roots: lmdeploy={lm_roots}, sglang={sg_roots}")
    reports = []
    for rank, (lm_root, sg_root) in enumerate(zip(lm_roots, sg_roots)):
        lm_rec, sg_rec = records(lm_root), records(sg_root)
        lm_ids = load(lm_root, lm_rec, "selector.candidate_ids").astype(np.int64)
        sg_ids = load(sg_root, sg_rec, "selector.candidate_ids").astype(np.int64)
        lm_scores = load(lm_root, lm_rec, "selector.unary_scores").astype(np.float64)
        sg_scores = load(sg_root, sg_rec, "selector.unary_scores").astype(np.float64)
        aligned_lm, aligned_sg = [], []
        rows = []
        for row in range(lm_ids.shape[0]):
            lm_map = {as_int(token): as_float(score) for token, score in zip(lm_ids[row], lm_scores[row])}
            sg_map = {as_int(token): as_float(score) for token, score in zip(sg_ids[row], sg_scores[row])}
            common = sorted(lm_map.keys() & sg_map.keys())
            for token in common:
                aligned_lm.append(lm_map[token])
                aligned_sg.append(sg_map[token])
            rows.append({
                "row": row,
                "common": len(common),
                "lm_only": sorted(lm_map.keys() - sg_map.keys()),
                "sg_only": sorted(sg_map.keys() - lm_map.keys()),
            })
        x, y = np.asarray(aligned_lm), np.asarray(aligned_sg)
        design = np.column_stack((x, np.ones_like(x)))
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        reports.append({
            "rank": rank,
            "aligned": as_int(x.size),
            "slope": as_float(slope),
            "intercept": as_float(intercept),
            "rms_raw": as_float(np.sqrt(np.mean((x - y) ** 2))),
            "rms_fit": as_float(np.sqrt(np.mean((x * slope + intercept - y) ** 2))),
            "mean_lm": as_float(x.mean()),
            "mean_sg": as_float(y.mean()),
            "rows": rows,
        })
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    print("DFLASH_SELECTOR_PARITY_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
