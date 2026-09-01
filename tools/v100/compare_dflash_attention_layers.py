#!/usr/bin/env python3
"""Compare native TP4 DFlash attention boundaries across all five layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DTYPES = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bf16": np.dtype("<u2"),
}


def rank_directories(root: Path) -> list[Path]:
    result = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
    if len(result) != 4:
        raise RuntimeError(f"expected four rank directories below {root}, got {result}")
    return result


def records(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line_number, line in enumerate((root / "manifest.jsonl").read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid manifest JSON at {root}:{line_number}") from error
        result.setdefault(record["name"], record)
    return result


def load(root: Path, manifest: dict[str, dict], name: str) -> np.ndarray:
    record = manifest[name]
    dtype = record["dtype"]
    value = np.fromfile(root / record["file"], dtype=DTYPES[dtype])
    if dtype == "bf16":
        value = (value.astype(np.uint32) << 16).view(np.float32)
    return value.reshape(record["shape"])


def rope_interleaved(value: np.ndarray) -> np.ndarray:
    shape = value.shape
    if shape[-1] % 128:
        raise RuntimeError(f"RoPE width is not head-dim-128 aligned: {shape}")
    return value.reshape(-1, shape[-1] // 128, 2, 64).transpose(0, 1, 3, 2).reshape(shape)


def statistics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool | list[int]]:
    if left.size != right.size:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    delta = left64 - right64
    try:
        different = int(np.count_nonzero(delta))
        max_abs = float(np.max(np.abs(delta), initial=0.0))
        rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    except (FloatingPointError, TypeError, ValueError) as error:
        raise RuntimeError("failed to reduce attention parity statistics") from error
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(left64, right64)),
        "allclose_f16": bool(np.allclose(left64, right64, rtol=2e-3, atol=2e-3)),
        "different": different,
        "max_abs": max_abs,
        "rms": rms,
    }


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    q32 = q.reshape(8, 8, 128).astype(np.float32)
    k32 = k.reshape(-1, 2, 128).astype(np.float32)
    v32 = v.reshape(-1, 2, 128).astype(np.float32)
    output = np.empty_like(q32)
    history = k32.shape[0] - q32.shape[0]
    for head in range(8):
        kv_head = head // 4
        scores = q32[:, head] @ k32[:, kv_head].T * np.float32(128.0**-0.5)
        if causal:
            for query in range(8):
                scores[query, history + query + 1 :] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        probability = np.exp(scores)
        probability /= probability.sum(axis=1, keepdims=True)
        output[:, head] = probability @ v32[:, kv_head]
    return output


def partial_attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q32 = q.reshape(8, 8, 128).astype(np.float32)
    k32 = k.reshape(-1, 2, 128).astype(np.float32)
    v32 = v.reshape(-1, 2, 128).astype(np.float32)
    partial_o = np.zeros((40, 16, 8, 128), dtype=np.float32)
    partial_lse = np.full((40, 16, 8), -(2**30), dtype=np.float32)
    active_splits = min(40, max(1, (k32.shape[0] + 127) // 128))
    split_len = (k32.shape[0] + active_splits - 1) // active_splits
    for split_id in range(active_splits):
        start = split_id * split_len
        end = min(start + split_len, k32.shape[0])
        for head in range(8):
            kv_head = head // 4
            scores = q32[:, head] @ k32[start:end, kv_head].T * np.float32(128.0**-0.5)
            maximum = scores.max(axis=1, keepdims=True)
            exponential = np.exp(scores - maximum)
            denominator = exponential.sum(axis=1, keepdims=True)
            partial_o[split_id, :8, head] = exponential @ v32[start:end, kv_head] / denominator
            partial_lse[split_id, :8, head] = np.log2(denominator[:, 0]) + maximum[:, 0] * np.log2(np.e)
    return partial_o, partial_lse


def combine_partials(partial_o: np.ndarray, partial_lse: np.ndarray, key_count: int) -> np.ndarray:
    active_splits = min(40, max(1, (key_count + 127) // 128))
    lse = partial_lse[:active_splits, :8].astype(np.float32)
    maximum = lse.max(axis=0, keepdims=True)
    weight = np.exp2(lse - maximum)
    weight /= weight.sum(axis=0, keepdims=True)
    return np.sum(weight[..., None] * partial_o[:active_splits, :8].astype(np.float32), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdeploy", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lm_roots = rank_directories(args.lmdeploy)
    sg_roots = rank_directories(args.sglang)
    rows: list[dict] = []
    for rank, (lm_root, sg_root) in enumerate(zip(lm_roots, sg_roots, strict=True)):
        lm = records(lm_root)
        sg = records(sg_root)
        for layer in range(5):
            prefix = f"layer{layer}.attention"
            required_lm = {
                f"{prefix}.qkv_projection",
                f"{prefix}.qkv_post_process",
                f"{prefix}.flattened_kv",
                f"{prefix}.core_output",
                f"{prefix}.tilelang.packed_q",
                f"{prefix}.tilelang.packed_k",
                f"{prefix}.tilelang.packed_v",
                f"{prefix}.tilelang.partial_o",
                f"{prefix}.tilelang.partial_lse",
                f"{prefix}.tilelang.cache_seqlens",
                f"{prefix}.tilelang.query_start_loc",
            }
            required_sg = {
                f"{prefix}.qkv_projection",
                f"{prefix}.backend_q",
                f"{prefix}.backend_k",
                f"{prefix}.backend_v",
                f"{prefix}.tilelang.q",
                f"{prefix}.tilelang.k",
                f"{prefix}.tilelang.v",
                f"{prefix}.tilelang.output",
            }
            if not required_lm <= lm.keys() or not required_sg <= sg.keys():
                raise RuntimeError(
                    f"rank {rank} layer {layer} missing LM={sorted(required_lm - lm.keys())} "
                    f"SG={sorted(required_sg - sg.keys())}"
                )

            lm_projection = load(lm_root, lm, f"{prefix}.qkv_projection").reshape(8, 1536)
            lm_post = load(lm_root, lm, f"{prefix}.qkv_post_process").reshape(8, 1536)
            sg_projection = load(sg_root, sg, f"{prefix}.qkv_projection").reshape(8, 1536)
            sg_backend_q = load(sg_root, sg, f"{prefix}.backend_q").reshape(8, 1024)
            sg_backend_k = load(sg_root, sg, f"{prefix}.backend_k").reshape(8, 2, 128)
            sg_backend_v = load(sg_root, sg, f"{prefix}.backend_v").reshape(8, 2, 128)
            sg_q = load(sg_root, sg, f"{prefix}.tilelang.q").reshape(8, 1024)
            sg_k = load(sg_root, sg, f"{prefix}.tilelang.k").reshape(-1, 2, 128)
            sg_v = load(sg_root, sg, f"{prefix}.tilelang.v").reshape(-1, 2, 128)
            sg_output = load(sg_root, sg, f"{prefix}.tilelang.output").reshape(8, 1024)
            key_count = sg_k.shape[0]
            lm_packed_q = load(lm_root, lm, f"{prefix}.tilelang.packed_q").reshape(8, 1024)
            lm_packed_k = load(lm_root, lm, f"{prefix}.tilelang.packed_k").reshape(key_count, 2, 128)
            lm_packed_v = load(lm_root, lm, f"{prefix}.tilelang.packed_v").reshape(key_count, 2, 128)
            lm_partial_o = load(lm_root, lm, f"{prefix}.tilelang.partial_o").reshape(40, 16, 8, 128)
            lm_partial_lse = load(lm_root, lm, f"{prefix}.tilelang.partial_lse").reshape(40, 16, 8)
            cache_seqlens = load(lm_root, lm, f"{prefix}.tilelang.cache_seqlens").reshape(-1)
            query_start_loc = load(lm_root, lm, f"{prefix}.tilelang.query_start_loc").reshape(-1)
            if not np.array_equal(cache_seqlens, np.array([key_count], dtype=np.int32)):
                raise AssertionError(f"rank {rank} layer {layer} cache_seqlens={cache_seqlens}")
            if not np.array_equal(query_start_loc, np.array([0, 8], dtype=np.int32)):
                raise AssertionError(f"rank {rank} layer {layer} query_start_loc={query_start_loc}")
            lm_core = load(lm_root, lm, f"{prefix}.core_output").reshape(8, 8, 128)
            sg_core = sg_output.reshape(8, 8, 128)
            noncausal_reference = attention_reference(sg_q, sg_k, sg_v, False)
            causal_reference = attention_reference(sg_q, sg_k, sg_v, True)
            partial_o_reference, partial_lse_reference = partial_attention_reference(sg_q, sg_k, sg_v)
            recombined_lm = combine_partials(lm_partial_o, lm_partial_lse, key_count)

            pairs = {
                "q_projection": (lm_projection[:, :1024], rope_interleaved(sg_projection[:, :1024])),
                "k_projection": (lm_projection[:, 1024:1280], rope_interleaved(sg_projection[:, 1024:1280])),
                "v_projection": (lm_projection[:, 1280:1536], sg_projection[:, 1280:1536]),
                "lm_q_post_vs_pre": (lm_post[:, :1024], lm_projection[:, :1024]),
                "lm_k_post_vs_pre": (lm_post[:, 1024:1280], lm_projection[:, 1024:1280]),
                "lm_v_post_vs_pre": (lm_post[:, 1280:1536], lm_projection[:, 1280:1536]),
                "sg_backend_q_vs_tilelang_q": (sg_backend_q, sg_q),
                "q_rotated": (lm_packed_q, sg_backend_q),
                "proposal_k": (lm_packed_k[-8:], sg_k[-8:]),
                "proposal_v": (lm_packed_v[-8:], sg_v[-8:]),
                "sg_cached_proposal_k_vs_backend": (sg_k[-8:], sg_backend_k),
                "sg_cached_proposal_v_vs_backend": (sg_v[-8:], sg_backend_v),
                "sg_cached_proposal_k_vs_zero": (sg_k[-8:], np.zeros_like(sg_k[-8:])),
                "sg_cached_proposal_v_vs_zero": (sg_v[-8:], np.zeros_like(sg_v[-8:])),
                "cache_k_prefix": (lm_packed_k[:1000], sg_k[:1000]),
                "cache_v_prefix": (lm_packed_v[:1000], sg_v[:1000]),
                "core_output": (lm_core, sg_core),
                "partial_o_vs_reference": (lm_partial_o[:8, :8], partial_o_reference[:8, :8]),
                "partial_lse_vs_reference": (lm_partial_lse[:8, :8], partial_lse_reference[:8, :8]),
                "lm_core_vs_recombined_partials": (lm_core, recombined_lm),
                "sg_core_vs_reference_noncausal": (sg_core, noncausal_reference),
                "sg_core_vs_reference_causal": (sg_core, causal_reference),
                "lm_core_vs_reference_noncausal": (lm_core, noncausal_reference),
                "lm_core_vs_reference_causal": (lm_core, causal_reference),
            }
            for boundary, (left, right) in pairs.items():
                row = {"rank": rank, "layer": layer, "boundary": boundary, **statistics(left, right)}
                rows.append(row)
                print("DFLASH_ATTENTION_LAYER_BOUNDARY " + json.dumps(row, sort_keys=True))
            if layer == 0:
                for split_id in range(8):
                    candidates = [
                        statistics(lm_partial_o[split_id, :8], partial_o_reference[reference_id, :8])["rms"]
                        for reference_id in range(8)
                    ]
                    try:
                        best_reference_split = int(np.argmin(candidates))
                    except (FloatingPointError, TypeError, ValueError) as error:
                        raise RuntimeError("failed to rank TileLang partial split references") from error
                    split_row = {
                        "rank": rank,
                        "layer": layer,
                        "boundary": "partial_o_split_assignment",
                        "split": split_id,
                        "best_reference_split": best_reference_split,
                        "candidate_rms": candidates,
                    }
                    rows.append(split_row)
                    print("DFLASH_ATTENTION_LAYER_BOUNDARY " + json.dumps(split_row, sort_keys=True))

    result = {"comparisons": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DFLASH_ATTENTION_LAYER_COMPARISON_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
