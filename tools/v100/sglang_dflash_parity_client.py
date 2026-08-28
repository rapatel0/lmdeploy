#!/usr/bin/env python3
"""Submit the audited 1K token-id prompt to a local SGLang trace server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def load_prompt_builder(path: Path):
    spec = importlib.util.spec_from_file_location("lmdeploy_bench_decode", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sglang_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt-builder", type=Path, required=True)
    parser.add_argument("--expected-hash", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    class Wrapper:
        pass

    outer = Wrapper()
    inner = Wrapper()
    inner.model = tokenizer
    outer.model = inner
    prompt, input_ids, prompt_hash = load_prompt_builder(args.prompt_builder)(outer, args.corpus, 1000)
    del prompt
    if prompt_hash != args.expected_hash:
        raise RuntimeError(f"prompt hash mismatch: {prompt_hash} != {args.expected_hash}")
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read()
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SGLang returned invalid JSON: {body[:500]!r}") from exc
    args.output.write_text(json.dumps({"prompt_hash": prompt_hash, "response": result}, indent=2))
    print(f"SGLANG_DFLASH_PARITY_REQUEST_PASS prompt_hash={prompt_hash}")


if __name__ == "__main__":
    main()
