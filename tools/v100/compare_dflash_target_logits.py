#!/usr/bin/env python3
"""Compare correlated prompt-frontier target logits across TP4 runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DTYPES = {"f16": np.dtype("<f2"), "f32": np.dtype("<f4")}


def rank_directories(root: Path) -> list[Path]:
    directories = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
    if len(directories) != 4:
        raise RuntimeError(f"expected four rank directories below {root}, got {directories}")
    return directories


def load_logits(directory: Path) -> tuple[np.ndarray, dict]:
    records: dict[str, dict] = {}
    for line_number, line in enumerate((directory / "manifest.jsonl").read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(f"invalid manifest JSON at {directory}:{line_number}") from error
        name = record["name"]
        if name in records:
            raise RuntimeError(f"duplicate manifest record {name!r} at {directory}:{line_number}")
        records[name] = record
    record = records["target.next_token_logits"]
    if record["dtype"] not in DTYPES or record["shape"] != [1, 248320]:
        raise RuntimeError(f"invalid target logits record in {directory}: {record}")
    payload = (directory / record["file"]).read_bytes()
    try:
        expected = int(np.prod(record["shape"])) * DTYPES[record["dtype"]].itemsize
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid target logits metadata in {directory}: {record}") from error
    if len(payload) != expected or record["bytes"] != expected:
        raise RuntimeError(f"invalid target logits payload in {directory}: {len(payload)} != {expected}")
    logits = np.frombuffer(payload, dtype=DTYPES[record["dtype"]]).astype(np.float32)
    return logits, record


def top_tokens(logits: np.ndarray, count: int = 16) -> list[dict[str, float | int]]:
    try:
        token_ids = np.arange(logits.size, dtype=np.int64)
        order = np.lexsort((token_ids, -logits))[:count]
        return [{"token_id": int(token), "logit": float(logits[token])} for token in order]
    except (FloatingPointError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("failed to rank target logits") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_logits: dict[str, np.ndarray] = {}
    rank_hashes: dict[str, list[str]] = {}
    metadata: dict[str, list[dict]] = {}
    for runtime, root in (("lmdeploy", args.lmdeploy), ("sglang", args.sglang)):
        values: list[np.ndarray] = []
        hashes: list[str] = []
        records: list[dict] = []
        for directory in rank_directories(root):
            logits, record = load_logits(directory)
            values.append(logits)
            hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
            records.append(record)
        if not all(np.array_equal(values[0], value) for value in values[1:]):
            raise RuntimeError(f"{runtime} target logits differ across TP ranks")
        runtime_logits[runtime] = values[0]
        rank_hashes[runtime] = hashes
        metadata[runtime] = records

    lm = runtime_logits["lmdeploy"]
    sg = runtime_logits["sglang"]
    delta = lm.astype(np.float64) - sg.astype(np.float64)
    lm_top = top_tokens(lm)
    sg_top = top_tokens(sg)
    try:
        different = int(np.count_nonzero(delta))
        max_abs = float(np.max(np.abs(delta), initial=0.0))
        rms = float(np.sqrt(np.mean(delta * delta)))
    except (FloatingPointError, TypeError, ValueError) as error:
        raise RuntimeError("failed to reduce target logit parity statistics") from error
    report = {
        "lmdeploy_top16": lm_top,
        "sglang_top16": sg_top,
        "top1_match": lm_top[0]["token_id"] == sg_top[0]["token_id"],
        "lmdeploy_top1": lm_top[0]["token_id"],
        "sglang_top1": sg_top[0]["token_id"],
        "different": different,
        "max_abs": max_abs,
        "rms": rms,
        "rank_hashes": rank_hashes,
        "records": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_LOGIT_PARITY_RESULT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
