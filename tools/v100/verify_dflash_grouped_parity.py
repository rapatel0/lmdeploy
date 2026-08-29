#!/usr/bin/env python3
"""Validate every TP-rank/full-layer grouped Q8H4 parity report."""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def fail(message: str) -> None:
    print(f"GROUPED_Q8H4_PARITY_FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir")
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--expected-layers-per-device", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.report_dir, "grouped-q8h4-pid-*-device-*.jsonl")))
    if not paths:
        fail("no report files")

    records = []
    for path in paths:
        with open(path, encoding="utf-8") as src:
            for line_number, line in enumerate(src, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    fail(f"{path}:{line_number}: {exc}")

    expected_total = args.expected_devices * args.expected_layers_per_device
    if len(records) != expected_total:
        fail(f"expected {expected_total} records, found {len(records)}")

    keys = Counter((record.get("device"), record.get("layer")) for record in records)
    duplicates = [key for key, count in keys.items() if count != 1]
    if duplicates:
        fail(f"duplicate device/layer records: {duplicates}")

    per_device = defaultdict(set)
    for record in records:
        per_device[record["device"]].add(record["layer"])
        required = {
            "queries": 8,
            "local_q_heads": 6,
            "local_kv_heads": 1,
            "cta_q": 8,
            "cta_h": 4,
            "base_ctas": 2,
            "splits": 8,
            "launched_ctas": 16,
            "elements": 8 * 6 * 256,
            "finite": True,
            "pass": True,
        }
        for name, expected in required.items():
            if record.get(name) != expected:
                fail(f"device={record.get('device')} layer={record.get('layer')} {name}={record.get(name)!r}, expected {expected!r}")
        if record["max_abs"] > record["max_abs_limit"]:
            fail(f"device={record['device']} layer={record['layer']} max_abs exceeded")
        if record["rms"] > record["rms_limit"]:
            fail(f"device={record['device']} layer={record['layer']} RMS exceeded")

    if len(per_device) != args.expected_devices:
        fail(f"expected {args.expected_devices} devices, found {sorted(per_device)}")
    layer_sets = list(per_device.values())
    if any(len(layers) != args.expected_layers_per_device for layers in layer_sets):
        fail(f"layer counts by device: {dict((device, len(layers)) for device, layers in per_device.items())}")
    if any(layers != layer_sets[0] for layers in layer_sets[1:]):
        fail("full-attention layer IDs differ across devices")

    print(
        "GROUPED_Q8H4_PARITY_PASS "
        f"records={len(records)} devices={len(per_device)} layers_per_device={len(layer_sets[0])} "
        f"max_abs={max(record['max_abs'] for record in records)} "
        f"max_rms={max(record['rms'] for record in records)}"
    )


if __name__ == "__main__":
    main()
