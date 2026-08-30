#!/usr/bin/env python3
"""Analyze the DFlash post-rollback attention-metadata A/B."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
ARMS = ("legacy", "rebuild")
COMMIT_RE = re.compile(r"final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps")
CANDIDATE_RE = re.compile(
    r"DFLASH_METADATA_CANDIDATES device=(\d+) phase=(\d+) uid=(\d+) frontier=(\d+) rebuild=(\d+) ids=([0-9,]+)"
)
REBUILD_RE = re.compile(
    r"DFLASH_METADATA_REBUILD_ACTIVE device=(\d+) phase=(\d+) rows=(\d+) block=(\d+) "
    r"old_q_sum=(\d+) old_k_sum=(\d+) new_q_sum=(\d+) new_k_sum=(\d+) assert=(\d+)"
)
OFFSET_ASSERT_RE = re.compile(
    r"DFLASH_METADATA_OFFSETS_ASSERT_ACTIVE device=(\d+) phase=(\d+) rows=(\d+) first_span=(\d+)"
)


def fail(message: str) -> None:
    raise SystemExit(f"DFLASH_METADATA_REBUILD_ANALYSIS_FAIL: {message}")


def arm(name: str) -> dict[str, object]:
    try:
        data = json.loads((ROOT / f"{name}.json").read_text())
        text = (ROOT / f"{name}.log").read_text(errors="replace")
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{name}: invalid artifacts: {exc}")
    trials = data.get("trials", [])
    if len(trials) != 5 or any(row.get("degenerate") or row.get("output_tokens") != 256 for row in trials):
        fail(f"{name}: missing or degenerate five-trial benchmark")
    commits = COMMIT_RE.findall(text)
    if not commits:
        fail(f"{name}: missing acceptance accounting")
    commit_length = statistics.median(float(row[0]) for row in commits)
    raw_length = statistics.median(float(row[1]) for row in commits)
    decode = float(data["mean_decode_tok_s"])
    candidates: dict[int, list[tuple[int, tuple[int, ...]]]] = {device: [] for device in range(4)}
    for device, _phase, _uid, frontier, rebuild, ids in CANDIDATE_RE.findall(text):
        if int(rebuild) != (name == "rebuild"):
            fail(f"{name}: candidate trace carried wrong route bit")
        candidates[int(device)].append((int(frontier), tuple(map(int, ids.split(",")))))
    if any(not rows for rows in candidates.values()):
        fail(f"{name}: missing four-rank candidate traces")
    return {
        "decode_tok_s": decode,
        "commit_length": commit_length,
        "raw_length": raw_length,
        "normalized_cycle_ms": 1000.0 * commit_length / decode,
        "candidates": candidates,
        "text": text,
    }


def main() -> None:
    arms = {name: arm(name) for name in ARMS}
    legacy_text = str(arms["legacy"].pop("text"))
    rebuild_text = str(arms["rebuild"].pop("text"))
    try:
        assert_text = (ROOT / "assert_smoke.log").read_text(errors="replace")
    except OSError as exc:
        fail(f"missing assertion smoke: {exc}")
    if "DFLASH_METADATA_REBUILD_ACTIVE" in legacy_text:
        fail("legacy arm unexpectedly rebuilt metadata")
    records = REBUILD_RE.findall(rebuild_text)
    devices = {int(row[0]) for row in records}
    if devices != {0, 1, 2, 3} or any(int(row[-1]) != 0 for row in records):
        fail(f"rebuild performance route incomplete devices={sorted(devices)} records={len(records)}")
    corrected = sum(int(row[5]) != int(row[7]) for row in records)
    if corrected == 0:
        fail("rebuild arm never changed a stale key-span sum")
    assert_records = REBUILD_RE.findall(assert_text)
    assert_devices = {int(row[0]) for row in assert_records if int(row[-1]) == 1}
    if assert_devices != {0, 1, 2, 3}:
        fail(f"metadata assertion route incomplete devices={sorted(assert_devices)}")
    offset_devices = {int(row[0]) for row in OFFSET_ASSERT_RE.findall(assert_text)}
    if offset_devices != {0, 1, 2, 3}:
        fail(f"live cu_k_len assertion route incomplete devices={sorted(offset_devices)}")

    comparison: dict[str, object] = {}
    exact = total = 0
    first_difference = None
    for device in range(4):
        grouped: dict[str, dict[int, list[tuple[int, ...]]]] = {}
        for name in ARMS:
            by_frontier: dict[int, list[tuple[int, ...]]] = defaultdict(list)
            for frontier, ids in arms[name]["candidates"][device]:
                by_frontier[frontier].append(ids)
            grouped[name] = by_frontier
        common = sorted(set(grouped["legacy"]) & set(grouped["rebuild"]))
        for frontier in common:
            for occurrence, (left, right) in enumerate(
                zip(grouped["legacy"][frontier], grouped["rebuild"][frontier])
            ):
                total += 1
                if left == right:
                    exact += 1
                elif first_difference is None:
                    first_difference = {
                        "device": device,
                        "frontier": frontier,
                        "occurrence": occurrence,
                        "legacy": left,
                        "rebuild": right,
                    }
    comparison["common_frontiers"] = total
    comparison["exact_candidate_blocks"] = exact
    comparison["candidate_exact_fraction"] = exact / total if total else 0.0
    comparison["first_candidate_difference"] = first_difference
    comparison["normalized_cycle_pct"] = 100.0 * (
        float(arms["rebuild"]["normalized_cycle_ms"]) / float(arms["legacy"]["normalized_cycle_ms"]) - 1.0
    )
    comparison["commit_length_delta"] = float(arms["rebuild"]["commit_length"]) - float(
        arms["legacy"]["commit_length"]
    )
    comparison["corrected_metadata_records"] = corrected
    for result in arms.values():
        result.pop("candidates")
    output = {"arms": arms, "comparison": comparison}
    (ROOT / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print("DFLASH_METADATA_REBUILD_ANALYSIS", json.dumps(output, sort_keys=True))
    print("DFLASH_METADATA_REBUILD_ROUTE_PASS", f"devices=4 corrected={corrected} common_frontiers={total}")


if __name__ == "__main__":
    main()
