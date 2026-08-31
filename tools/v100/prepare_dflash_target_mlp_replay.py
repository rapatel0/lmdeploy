#!/usr/bin/env python3
"""Extract SGLang's TP4 layer-0 MLP-normalized final prompt row."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

AUDITED_PROMPT_SHA256 = "9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01"
AUDITED_TOKEN_198_EMBEDDING_SHA256 = "19ae10cba7028c81d25b525c086479035b4069702bcd74764e16e66495669234"


def load_manifest(path: Path) -> list[dict[str, object]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read manifest {path}: {error}") from error


def rank_dirs(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in root.iterdir():
        manifest = directory / "manifest.jsonl"
        if not directory.is_dir() or not manifest.is_file():
            continue
        records = load_manifest(manifest)
        if records:
            try:
                result[int(records[0]["tp_rank"])] = directory
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid TP rank in {manifest}") from error
    if set(result) != {0, 1, 2, 3}:
        raise RuntimeError(f"expected TP4 SGLang trace, got {sorted(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance: list[dict[str, object]] = []
    for rank, directory in sorted(rank_dirs(args.sglang).items()):
        records = load_manifest(directory / "manifest.jsonl")
        matching = [record for record in records if record["name"] == "target.trajectory"]
        if len(matching) != 1:
            raise RuntimeError(f"rank {rank} missing target trajectory")
        record = matching[0]
        if record.get("position") != 999:
            raise RuntimeError(f"rank {rank} has invalid target alignment: {record}")
        try:
            trajectory_file = str(record["file"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"rank {rank} has invalid trajectory metadata") from error
        trajectory = np.fromfile(directory / trajectory_file, dtype="<f2").reshape(38, 5120)
        token_id = record.get("token_id")
        if token_id != 198:
            # Some split-prefill ForwardBatch objects omit input_ids. Accept
            # that explicit sentinel only when both the audited prompt hash
            # and the checkpoint-exact token-198 embedding prove alignment.
            embedding_sha256 = hashlib.sha256(trajectory[0].tobytes()).hexdigest()
            if not (
                token_id == -1
                and record.get("input_ids_sha256") == AUDITED_PROMPT_SHA256
                and embedding_sha256 == AUDITED_TOKEN_198_EMBEDDING_SHA256
            ):
                raise RuntimeError(f"rank {rank} has invalid target alignment: {record}")
        row = trajectory[4].copy()  # target.layer0.mlp_norm
        path = args.output / f"rank-{rank}.bin"
        row.tofile(path)
        activation_records = [record for record in records if record["name"] == "target.layer0.mlp_activation"]
        if len(activation_records) != 1:
            raise RuntimeError(f"rank {rank} missing target MLP activation")
        try:
            activation_file = str(activation_records[0]["file"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"rank {rank} has invalid target MLP activation metadata") from error
        activation = np.fromfile(directory / activation_file, dtype="<f2")
        if activation.size != 4352:
            raise RuntimeError(f"rank {rank} target MLP activation has {activation.size} values")
        activation_path = args.output / f"rank-{rank}-activation.bin"
        activation.tofile(activation_path)
        provenance.append(
            {
                "rank": rank,
                "source": str(directory),
                "position": record["position"],
                "token_id": record["token_id"],
                "input_ids_sha256": record.get("input_ids_sha256"),
                "bytes": path.stat().st_size,
                "activation_bytes": activation_path.stat().st_size,
            }
        )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("DFLASH_TARGET_MLP_REPLAY_PREPARED " + json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
