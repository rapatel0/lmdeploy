#!/usr/bin/env python3
"""Compare TurboMind's walked selector rows with SGLang's full lattice."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

DTYPES = {"f16": "<f2", "f32": "<f4", "i32": "<i4", "i64": "<i8"}


def roots(path: str) -> list[str]:
    found = sorted(glob.glob(path + "/rank-*-pid-*"))
    if len(found) != 4:
        raise RuntimeError(f"expected four roots under {path}, found {found}")
    return found


def records(root: str) -> dict[str, dict]:
    try:
        with open(Path(root, "manifest.jsonl"), encoding="utf-8") as manifest:
            return {row["name"]: row for row in map(json.loads, manifest)}
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"invalid manifest under {root}: {exc}") from exc


def load(root: str, rec: dict[str, dict], name: str) -> np.ndarray:
    row = rec[name]
    return np.fromfile(Path(root, row["file"]), dtype=DTYPES[row["dtype"]]).reshape(row["shape"])


def scalar(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"cannot convert {value!r} to float") from exc


def integer(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"cannot convert {value!r} to int") from exc


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "slope": scalar(slope),
        "intercept": scalar(intercept),
        "rms_raw": scalar(np.sqrt(np.mean((x - y) ** 2))),
        "rms_fit": scalar(np.sqrt(np.mean((x * slope + intercept - y) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", required=True)
    parser.add_argument("--sglang", required=True)
    args = parser.parse_args()
    reports = []
    for rank, (lm_root, sg_root) in enumerate(zip(roots(args.lmdeploy), roots(args.sglang))):
        lm_rec, sg_rec = records(lm_root), records(sg_root)
        lm_ids = load(lm_root, lm_rec, "selector.candidate_ids").reshape(7, 16)
        sg_ids = load(sg_root, sg_rec, "selector.candidate_ids").reshape(7, 16)
        selected = load(sg_root, sg_rec, "selector.selected_ids").reshape(7)
        lm_unary = load(lm_root, lm_rec, "selector.unary_scores").reshape(7, 16).astype(np.float64)
        sg_unary = load(sg_root, sg_rec, "selector.unary_scores").reshape(7, 16).astype(np.float64)
        lm_walked = load(lm_root, lm_rec, "selector.scores").reshape(7, 16).astype(np.float64)
        sg_lattice = load(sg_root, sg_rec, "selector.scores").reshape(7, 16, 16).astype(np.float64)
        score_lm, score_sg, unary_lm, unary_sg, transition_lm, transition_sg = [], [], [], [], [], []
        rows = []
        predecessor_index = 0
        for slot in range(7):
            sg_walked = sg_lattice[slot, predecessor_index]
            lm_map = {integer(token): index for index, token in enumerate(lm_ids[slot])}
            sg_map = {integer(token): index for index, token in enumerate(sg_ids[slot])}
            common = sorted(lm_map.keys() & sg_map.keys())
            for token in common:
                li, si = lm_map[token], sg_map[token]
                score_lm.append(lm_walked[slot, li])
                score_sg.append(sg_walked[si])
                unary_lm.append(lm_unary[slot, li])
                unary_sg.append(sg_unary[slot, si])
                transition_lm.append(lm_walked[slot, li] - lm_unary[slot, li])
                transition_sg.append(sg_walked[si] - sg_unary[slot, si])
            rows.append({"slot": slot, "common": len(common), "predecessor_index": predecessor_index})
            if slot < 6:
                matches = np.flatnonzero(sg_ids[slot] == selected[slot])
                if matches.size != 1:
                    raise RuntimeError(f"selected token {selected[slot]} not unique in slot {slot}")
                predecessor_index = integer(matches[0])
        arrays = {
            "score": (np.asarray(score_lm), np.asarray(score_sg)),
            "unary": (np.asarray(unary_lm), np.asarray(unary_sg)),
            "transition": (np.asarray(transition_lm), np.asarray(transition_sg)),
        }
        reports.append({"rank": rank, "rows": rows, **{name: fit(*values) for name, values in arrays.items()}})
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    print("DFLASH_SELECTOR_LATTICE_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
