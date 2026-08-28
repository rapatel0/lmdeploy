"""Load the separate DFlash2 checkpoint into TurboMind and run target-only text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def read_json(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'FAIL: cannot read {path}: {exc}') from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--tp', type=int, default=4)
    args = parser.parse_args()

    config = read_json(os.path.join(args.draft_model, 'config.json'))
    architectures = config.get('architectures', [])
    draft = config.get('dflash_config', {})
    print(f'architectures={architectures}')
    print(f'num_hidden_layers={config.get("num_hidden_layers")}')
    print(f'dflash_config={draft}')
    if 'DFlash2DraftModel' not in architectures:
        print('FAIL: draft checkpoint is not DFlash2DraftModel', file=sys.stderr)
        return 2
    required = {
        'block_size': 8,
        'conv_kernel_size': 2,
        'conv_group_size': 16,
        'selector_rank': 256,
        'selector_top_k': 16,
    }
    for name, expected in required.items():
        if draft.get(name) != expected:
            print(f'FAIL: {name}={draft.get(name)!r}, expected {expected}', file=sys.stderr)
            return 2
    if config.get('num_hidden_layers') != 5:
        print('FAIL: expected five draft layers', file=sys.stderr)
        return 2

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if '[DFlash2]' in message:
                records.append(message)

    logger = logging.getLogger('lmdeploy')
    handler = Capture()
    logger.addHandler(handler)
    engine = TurbomindEngineConfig(
        model_format='fp8',
        tp=args.tp,
        max_batch_size=1,
        cache_max_entry_count=0.05,
        async_=0,
        num_draft_tokens=0,
        speculative_algorithm='dflash2',
        speculative_draft_model=args.draft_model,
        speculative_dflash_block_size=8,
        speculative_draft_window=2048,
    )
    pipe = pipeline(args.model, backend_config=engine, log_level='INFO')
    try:
        output = pipe(
            ['Hi'],
            gen_config=GenerationConfig(max_new_tokens=24, temperature=0.0, do_sample=False),
        )[0].text.strip()
        print(f'output={output!r}')
    finally:
        pipe.close()
        logger.removeHandler(handler)

    for record in records:
        print(record)
    if not records:
        print('FAIL: loader emitted no DFlash2 weight record', file=sys.stderr)
        return 3
    if len(set(output.split())) < 3:
        print('FAIL: generated output is degenerate', file=sys.stderr)
        return 4
    print('DFLASH_LOAD_PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
