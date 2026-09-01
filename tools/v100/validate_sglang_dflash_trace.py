#!/usr/bin/env python3
"""Validate a source-audited TP4 SGLang DFlash verifier trace."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct

REQUIRED = {
    "target.post_layer_residual",
    "target.next_token_logits",
    "target.next_token_top16_ids",
    "target.next_token_top16_values",
    "context.fc",
    "context.norm",
    "block.ids",
    "block.embedding",
    "layer0.attention.conv_side0",
    "layer4.output.hidden",
    "selector.candidate_ids",
    "selector.unary_scores",
    "selector.score_lattice",
    "selector.selected_ids",
    "layer0.attention.tilelang.q",
    "layer0.attention.tilelang.k",
    "layer0.attention.tilelang.v",
    "layer0.attention.tilelang.output",
    "layer0.attention.tilelang.block_table",
    "layer0.attention.tilelang.seq_lens",
    "layer0.attention.tilelang.query_start_loc",
    "layer0.attention.tilelang.prefix_kv_lens",
}
for _layer in range(5):
    REQUIRED.update(
        {
            f"layer{_layer}.attention.qkv_projection",
            f"layer{_layer}.attention.tilelang.q",
            f"layer{_layer}.attention.tilelang.k",
            f"layer{_layer}.attention.tilelang.v",
            f"layer{_layer}.attention.tilelang.output",
        }
    )


def _integer_values(directory: pathlib.Path, record: dict) -> list[int]:
    payload = (directory / record["file"]).read_bytes()
    fmt = {"i32": "i", "i64": "q"}[record["dtype"]]
    width = struct.calcsize(fmt)
    if len(payload) % width:
        raise AssertionError(f"unaligned {record['name']} payload in {directory}")
    return list(struct.unpack("<" + fmt * (len(payload) // width), payload))


def _load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid JSON file {path}: {error}") from error


def _load_manifest(path: pathlib.Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
        return [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid manifest {path}: {error}") from error


def _validate_policy(policy: dict) -> None:
    backend = policy["backend"]
    assert backend in {"flash_attn_v100", "tilelang_fa_v100.paged_forward"}
    assert policy["linear_verify"]
    if backend == "flash_attn_v100":
        assert policy["target_verify"]
        assert policy["layer_id"] == 0
        assert policy["attention_type"] == "AttentionType.ENCODER_ONLY"
        # SGLang metadata starts causal, then the layer contract resolves the
        # encoder-only DFlash verifier to full-prefix non-causal attention.
        assert policy["metadata_causal"]
        assert not policy["resolved_causal"]
        assert policy["resolved_window_size"] == -1
        assert policy["query_dtype"] == "torch.float16"
        assert policy["key_cache_dtype"] == "torch.float16"
        assert policy["value_cache_dtype"] == "torch.float16"
    else:
        # This is the stronger capture point: values are recorded inside the
        # actual embedded TileLang call rather than reconstructed by its
        # FlashAttention backend caller.
        assert not policy["causal"]
    assert policy["block_size"] == 16
    assert math.isclose(policy["softmax_scale"], 0.08838834764831845)
    assert policy["query_shape"] == [8, 8, 128]
    assert policy["query_stride"] == [1024, 128, 1]
    assert policy["key_cache_shape"] == [1025, 16, 2, 128]
    assert policy["key_cache_stride"] == [4096, 256, 128, 1]
    assert policy["value_cache_shape"] == policy["key_cache_shape"]
    assert policy["value_cache_stride"] == policy["key_cache_stride"]
    assert policy["page_table_shape"] == [1, 1024]
    assert policy["page_table_stride"] == [1024, 1]
    assert policy["sequence_lengths"] == [1008]
    assert policy["query_start_locations"] == [0, 8]
    assert policy["prefix_kv_lengths"] == [1000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--block-index", type=int, default=1)
    parser.add_argument("--expected-anchor", type=int)
    args = parser.parse_args()
    if args.block_index < 1:
        raise SystemExit("--block-index is 1-based")

    directories = sorted(path for path in args.root.glob("rank-*-pid-*") if path.is_dir())
    assert len(directories) == 4, f"expected four TP trace directories, got {directories}"
    policies: list[dict] = []
    audited_anchors: list[int] = []

    for directory in directories:
        records = _load_manifest(directory / "manifest.jsonl")
        names = {record["name"] for record in records}
        assert REQUIRED <= names, f"missing {sorted(REQUIRED - names)} from {directory}"

        by_name: dict[str, dict] = {}
        duplicates: set[str] = set()
        for record in records:
            assert record.get("capture_block_index") == args.block_index, record
            name = record["name"]
            if name in by_name:
                duplicates.add(name)
            by_name[name] = record
        assert not duplicates, f"duplicate trace records in {directory}: {sorted(duplicates)}"
        for layer in range(5):
            prefix = f"layer{layer}.attention.tilelang"
            for suffix, shape in (
                ("q", [8, 8, 128]),
                ("k", [1008, 2, 128]),
                ("v", [1008, 2, 128]),
                ("output", [8, 8, 128]),
            ):
                record = by_name[f"{prefix}.{suffix}"]
                assert record["dtype"] == "f16", record
                assert record["shape"] == shape, record

        logits = by_name["target.next_token_logits"]
        top_ids = by_name["target.next_token_top16_ids"]
        top_values = by_name["target.next_token_top16_values"]
        target_top_ids = _integer_values(directory, top_ids)
        expected_anchor = args.expected_anchor if args.expected_anchor is not None else target_top_ids[0]
        block_record = by_name["block.ids"]
        assert block_record.get("draft_block_index") == args.block_index, block_record
        block_ids = _integer_values(directory, block_record)
        expected_ids = [expected_anchor] + [248070] * 7
        assert block_ids == expected_ids, f"non-audited block IDs in {directory}: {block_ids} != {expected_ids}"
        audited_anchors.append(expected_anchor)
        assert logits["dtype"] in {"f16", "f32"} and logits["shape"] == [1, 248320], logits
        assert top_ids["dtype"] == "i64" and top_ids["shape"] == [16], top_ids
        assert top_values["dtype"] == "f32" and top_values["shape"] == [16], top_values

        assert _integer_values(directory, by_name["layer0.attention.tilelang.seq_lens"]) == [1008]
        assert _integer_values(directory, by_name["layer0.attention.tilelang.query_start_loc"]) == [0, 8]
        assert _integer_values(directory, by_name["layer0.attention.tilelang.prefix_kv_lens"]) == [1000]

        policy_path = directory / "tilelang_policy.json"
        assert policy_path.is_file(), f"missing TileLang policy audit in {directory}"
        policy = _load_json(policy_path)
        _validate_policy(policy)
        policies.append(policy)

        for record in records:
            data = directory / record["file"]
            assert data.stat().st_size == record["bytes"]

    canonical = {json.dumps(policy, sort_keys=True) for policy in policies}
    assert len(canonical) == 1, f"TileLang policy differs across TP ranks: {policies}"
    assert len(set(audited_anchors)) == 1, f"target anchors differ across TP ranks: {audited_anchors}"
    print(f"SGLANG_DFLASH_TILELANG_POLICY_PASS {canonical.pop()}")
    print(
        f"SGLANG_DFLASH_PARITY_TRACE_PASS ranks={len(directories)} "
        f"block_index={args.block_index} anchor={audited_anchors[0]}"
    )


if __name__ == "__main__":
    main()
