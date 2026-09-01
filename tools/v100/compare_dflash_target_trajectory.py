#!/usr/bin/env python3
"""Compare the first audited target trajectory across TP4 runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def boundary_names() -> list[str]:
    names = ["target.input.embedding", "target.layer0.attn_norm"]
    for layer in range(6):
        names.extend(
            [
                f"target.layer{layer}.branch",
                f"target.layer{layer}.post_attn_residual",
                f"target.layer{layer}.mlp_norm",
                f"target.layer{layer}.mlp_output",
                f"target.layer{layer}.output_residual",
                f"target.layer{layer}.next_attn_norm",
            ]
        )
    return names


def rank_dirs(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    try:
        directories = list(root.iterdir())
        for directory in directories:
            if not directory.is_dir() or not (directory / "manifest.jsonl").is_file():
                continue
            records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
            if not records:
                continue
            rank = int(records[0]["tp_rank"])
            if rank in result:
                raise RuntimeError(f"duplicate TP rank {rank} under {root}")
            result[rank] = directory
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read trajectory manifests under {root}: {error}") from error
    if set(result) != {0, 1, 2, 3}:
        raise RuntimeError(f"expected TP ranks 0..3 under {root}, got {sorted(result)}")
    return result


def load(directory: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    try:
        records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read trajectory manifest in {directory}: {error}") from error
    matching = [record for record in records if record["name"] == "target.trajectory"]
    if len(matching) != 1:
        raise RuntimeError(f"expected one target.trajectory in {directory}, got {len(matching)}")
    record = matching[0]
    if record["dtype"] != "f16" or record["shape"] != [38, 5120]:
        raise RuntimeError(f"unexpected trajectory contract in {directory}: {record}")
    value = np.fromfile(directory / record["file"], dtype="<f2").reshape(38, 5120)
    dtype_matching = [record for record in records if record["name"] == "target.trajectory_dtypes"]
    dtype_codes = None
    if dtype_matching:
        dtype_record = dtype_matching[0]
        dtype_codes = np.fromfile(directory / dtype_record["file"], dtype="<i4")
        if dtype_codes.shape != (38,):
            raise RuntimeError(f"unexpected target dtype codes in {directory}: {dtype_codes.shape}")
    metadata = dict(record)
    ids_matching = [item for item in records if item["name"] == "target.input_ids"]
    if ids_matching:
        ids_record = ids_matching[0]
        ids = np.fromfile(directory / ids_record["file"], dtype="<i4")
        metadata["input_ids_sha256"] = hashlib.sha256(ids[:1000].tobytes(order="C")).hexdigest()
        try:
            metadata["input_rows"] = int(ids.size)
            if ids.size >= 1000:
                metadata["input_token_id"] = int(ids[999])
        except (TypeError, ValueError, IndexError) as error:
            raise RuntimeError(f"invalid target input metadata in {directory}") from error
    return value, dtype_codes, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lm_dirs = rank_dirs(args.lmdeploy)
    sg_dirs = rank_dirs(args.sglang)
    names = boundary_names()
    rows: list[dict[str, object]] = []
    earliest = None
    for rank in range(4):
        lm, _, lm_metadata = load(lm_dirs[rank])
        sg, sg_dtype_codes, sg_metadata = load(sg_dirs[rank])
        if lm_metadata.get("position") != 999 or sg_metadata.get("position") != 999:
            raise RuntimeError(
                f"rank {rank} target position mismatch: "
                f"lm={lm_metadata.get('position')} sg={sg_metadata.get('position')} expected=999"
            )
        if lm_metadata.get("input_ids_sha256") != sg_metadata.get("input_ids_sha256"):
            raise RuntimeError(f"rank {rank} target prompt input hash mismatch")
        # Early SGLang trajectory captures wrote token_id=-1 before the capture
        # hook could observe the final prompt token and embedded only the audited
        # input hash in trajectory metadata. Resolve that legacy form only for
        # the exact audited prompt; newer records must carry input_ids directly.
        expected_prompt_hash = "9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01"
        lm_token = lm_metadata.get("input_token_id")
        sg_token = sg_metadata.get("input_token_id")
        if (
            sg_token is None
            and sg_metadata.get("token_id") == -1
            and sg_metadata.get("input_ids_sha256") == expected_prompt_hash
        ):
            sg_token = 198
        if lm_token != 198 or sg_token != 198:
            raise RuntimeError(
                f"rank {rank} target token mismatch: lm={lm_token} sg={sg_token} expected=198"
            )
        if lm_metadata.get("token_id") not in (198, -1) or sg_metadata.get("token_id") not in (198, -1):
            raise RuntimeError(
                f"rank {rank} invalid trajectory token metadata: "
                f"lm={lm_metadata.get('token_id')} sg={sg_metadata.get('token_id')}"
            )
        for index, name in enumerate(names):
            lhs = lm[index].astype(np.float32)
            rhs = sg[index].astype(np.float32)
            try:
                lhs_finite = bool(np.isfinite(lhs).all())
                rhs_finite = bool(np.isfinite(rhs).all())
                if not lhs_finite or not rhs_finite:
                    row = {
                        "rank": rank,
                        "index": index,
                        "name": name,
                        "status": "nonfinite",
                        "lmdeploy_nonfinite": int((~np.isfinite(lhs)).sum()),
                        "sglang_nonfinite": int((~np.isfinite(rhs)).sum()),
                        "sglang_source_dtype": int(sg_dtype_codes[index]) if sg_dtype_codes is not None else None,
                    }
                    differing = 1
                else:
                    delta = lhs - rhs
                    differing = int(np.count_nonzero(lm[index].view(np.uint16) != sg[index].view(np.uint16)))
                    row = {
                        "rank": rank,
                        "index": index,
                        "name": name,
                        "status": "mismatch" if differing else "match",
                        "differing": differing,
                        "max_abs": float(np.max(np.abs(delta))),
                        "rms": float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
                        "first_coordinate": int(np.flatnonzero(delta)[0]) if differing else None,
                        "sglang_source_dtype": int(sg_dtype_codes[index]) if sg_dtype_codes is not None else None,
                    }
                if differing and (earliest is None or index < int(earliest["index"])):
                    earliest = row
            except (ValueError, TypeError, IndexError, KeyError) as error:
                raise RuntimeError(f"cannot compare rank {rank} boundary {name}: {error}") from error
            rows.append(row)

    result = {"earliest": earliest, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if earliest is None:
        print("DFLASH_TARGET_TRAJECTORY_MATCH")
    else:
        print("DFLASH_TARGET_TRAJECTORY_FIRST " + json.dumps(earliest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
