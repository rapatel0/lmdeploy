#!/usr/bin/env python3
"""Measure one exact-length SGLang request and retain acceptance metadata."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def build_input_ids(tokenizer, target_tokens: int, nonce: int) -> list[int]:
    filler = tokenizer.encode(
        f"Benchmark nonce {nonce}. The quick brown fox crosses the valley while "
        "a careful engineer traces model state and verifies every result. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("the tokenizer returned an empty filler")
    return (filler * ((target_tokens + len(filler) - 1) // len(filler)))[:target_tokens]


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"expected a numeric benchmark field, received {value!r}") from exc


def run_request(
    base_url: str,
    input_ids: list[int],
    output_tokens: int,
) -> dict[str, object]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_new_tokens": output_tokens,
            "min_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    started = time.perf_counter()
    first_at = None
    final: dict[str, Any] = {}
    with requests.post(f"{base_url}/generate", json=payload, stream=True, timeout=900) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid SGLang stream event: {line[:200]!r}") from exc
            if not isinstance(event, dict):
                raise RuntimeError(f"unexpected SGLang stream event: {event!r}")
            final = event
            meta = event.get("meta_info") or {}
            if not isinstance(meta, dict):
                raise RuntimeError(f"unexpected SGLang metadata: {meta!r}")
            if first_at is None and as_int(meta.get("completion_tokens")) > 0:
                first_at = time.perf_counter()
    finished = time.perf_counter()
    if first_at is None:
        raise RuntimeError("the streaming response produced no completion token")
    meta = final.get("meta_info") or {}
    if not isinstance(meta, dict):
        raise RuntimeError(f"unexpected final SGLang metadata: {meta!r}")
    completion_tokens = as_int(meta.get("completion_tokens"))
    if completion_tokens != output_tokens:
        raise RuntimeError(f"expected {output_tokens} completion tokens, received {completion_tokens}")
    decode_seconds = finished - first_at
    return {
        "prompt_tokens": as_int(meta.get("prompt_tokens"), len(input_ids)),
        "completion_tokens": completion_tokens,
        "ttft_ms": round((first_at - started) * 1000, 3),
        "decode_seconds": round(decode_seconds, 6),
        "decode_tok_s": round((completion_tokens - 1) / decode_seconds, 3),
        "total_seconds": round(finished - started, 6),
        "text": str(final.get("text", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    warmup_ids = build_input_ids(tokenizer, args.input_tokens, 0)
    warmup = run_request(args.base_url, warmup_ids, args.output_tokens)
    print("warmup", json.dumps({k: v for k, v in warmup.items() if k != "text"}))

    trials = []
    for trial in range(1, args.trials + 1):
        row = run_request(
            args.base_url,
            build_input_ids(tokenizer, args.input_tokens, trial),
            args.output_tokens,
        )
        trials.append(row)
        print("trial", trial, json.dumps({k: v for k, v in row.items() if k != "text"}))

    server_info = requests.get(f"{args.base_url}/server_info", timeout=30).json()
    states = server_info.get("internal_states") or [{}]
    accept_length = states[0].get("avg_spec_accept_length")
    summary = {
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "mean_decode_tok_s": round(statistics.mean(as_float(row["decode_tok_s"]) for row in trials), 3),
        "avg_spec_accept_length": accept_length,
        "trials": trials,
        "server_info": server_info,
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k not in {"trials", "server_info"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
