#!/usr/bin/env python3
"""Compare DFlash proposal and acceptance traces for rejected-KV poisoning."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
proposal_re = re.compile(
    r"DFLASH_KV_PROPOSAL_TRACE poison=([01]) uid=\d+ tip=(\d+) ids=([0-9,-]+)"
)
accept_re = re.compile(
    r"final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps "
    r"\((\d+) committed, (\d+) raw committed, (\d+) accepted drafts, "
    r"(\d+) raw accepted drafts, (\d+) ambiguous steps, (\d+) tokens discarded by ambiguity, "
    r"(\d+) full accepts\)"
)


def load(arm: str, poison: int) -> tuple[list[tuple[str, str]], tuple[str, ...]]:
    text = (root / f"{arm}.log").read_text(errors="replace")
    # Four TP ranks emit the same records, but their host log lines can
    # interleave differently. Sort the complete multiset before comparison.
    proposals = sorted((tip, ids) for mode, tip, ids in proposal_re.findall(text) if int(mode) == poison)
    accepts = accept_re.findall(text)
    if not proposals:
        raise SystemExit(f"DFLASH_REJECTED_KV_POISON_FAIL {arm}: no proposal trace")
    if not accepts:
        raise SystemExit(f"DFLASH_REJECTED_KV_POISON_FAIL {arm}: no acceptance summary")
    return proposals, accepts[-1]


control_a, accept_a = load("control_a", 0)
poison, accept_poison = load("poison", 1)
control_b, accept_b = load("control_b", 0)
if control_a != control_b or accept_a != accept_b:
    raise SystemExit(
        "DFLASH_REJECTED_KV_POISON_INCONCLUSIVE: repeated controls differ "
        f"proposal_a={len(control_a)} proposal_b={len(control_b)} accept_a={accept_a} accept_b={accept_b}"
    )
if poison != control_a:
    raise SystemExit(
        "DFLASH_REJECTED_KV_POISON_FAIL: proposal multiset changed "
        f"control_count={len(control_a)} poison_count={len(poison)}"
    )
if accept_poison != accept_a:
    raise SystemExit(
        f"DFLASH_REJECTED_KV_POISON_FAIL: acceptance changed control={accept_a} poison={accept_poison}"
    )
result = {
    "proposal_records": len(poison),
    "acceptance": accept_poison,
    "candidate_match": True,
    "acceptance_match": True,
}
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print("DFLASH_REJECTED_KV_POISON_PASS", json.dumps(result, sort_keys=True))
