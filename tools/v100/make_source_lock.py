#!/usr/bin/env python3
"""Generate the immutable source lock for the lmdeploy-v100-next campaign.

The lock records machine facts only. It records no reviewed decision.

The tool fails closed. Any missing checkout, moved head, dirty worktree,
wrong campaign branch, or unresolved license produces a non-zero exit code.

The lock is immutable once written. The tool refuses to overwrite an existing
lock. Replacing an approved lock requires the explicit --relock flag, which
the master specification permits only at a phase boundary with operator
approval. A failed run never overwrites a good lock; it writes a diagnostic
file instead.

Usage:
    python3 tools/v100/make_source_lock.py --out docs/v100/source-lock.json
    python3 tools/v100/make_source_lock.py --out docs/v100/source-lock.json --relock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("~/repos/cust-lmdeploy").expanduser()

# The campaign branch named in the master specification. The generator refuses
# to record a branch claim it has not verified.
CAMPAIGN_BRANCH = "experiment/lmdeploy-v100-next"

APACHE_ROOT = "LICENSE file at repository root"

MARLIN_EVIDENCE = (
    "Per-file Apache-2.0 headers in csrc/. "
    "csrc/quantization/marlin/marlin.cu carries the Marlin and Neural Magic "
    "Apache-2.0 notice. No LICENSE file at repository root."
)

TILELANG_EVIDENCE = (
    "README.md License section states MIT. "
    "No LICENSE file at repository root."
)

# Requested refs are the approved audit anchors from the master specification.
#
# `match` selects the head rule:
#   ancestor -- the expected commit must be reachable from the current head.
#               The product base uses this, because the campaign branch adds
#               commits on top of the base.
#   head     -- the head must equal the expected commit exactly.
#               Every donor and reference source uses this, because those
#               checkouts must not move during the campaign.
SOURCES = [
    {
        "id": "A",
        "name": "lmdeploy",
        "path": "lmdeploy",
        "requested_ref": "v0.16.0",
        "expected_commit": "1208bf006bbac69f1f012ceafeeeb70f623b632c",
        "match": "ancestor",
        "role": "product_base",
        "license_files": ["LICENSE"],
        "spdx": "Apache-2.0",
        "license_evidence": APACHE_ROOT,
    },
    {
        "id": "B",
        "name": "zh-nj/lmdeploy-v100",
        "path": "lmdeploy-v100",
        "requested_ref": "d7c29f88e44016d7a757850fe761c1b1b66181c8",
        "expected_commit": "d7c29f88e44016d7a757850fe761c1b1b66181c8",
        "match": "head",
        "role": "donor",
        "license_files": ["LICENSE"],
        "spdx": "Apache-2.0",
        "license_evidence": APACHE_ROOT,
    },
    {
        "id": "C",
        "name": "1CatAI/1Cat-vLLM",
        "path": "1Cat-vLLM",
        "requested_ref": "675a12dedcca8cc020c033f1b1d0f1751d4b8efe",
        "expected_commit": "675a12dedcca8cc020c033f1b1d0f1751d4b8efe",
        "match": "head",
        "role": "donor",
        "license_files": ["LICENSE"],
        "spdx": "Apache-2.0",
        "license_evidence": APACHE_ROOT,
    },
    {
        "id": "D",
        "name": "haohervchb/sglang-V100",
        "path": "sglang-V100",
        "requested_ref": "0083b9fd83a601b1fcd9a691f7240be4e6be111e",
        "expected_commit": "0083b9fd83a601b1fcd9a691f7240be4e6be111e",
        "match": "head",
        "role": "donor",
        "license_files": ["LICENSE"],
        "spdx": "Apache-2.0",
        "license_evidence": APACHE_ROOT,
    },
    {
        "id": "E",
        "name": "haohervchb/Tilelang-FA-V100",
        "path": "Tilelang-FA-V100",
        "requested_ref": "c6332ebb0670efb7702eeb1b0e0d4477bea49def",
        "expected_commit": "c6332ebb0670efb7702eeb1b0e0d4477bea49def",
        "match": "head",
        "role": "reference",
        "license_files": [],
        "spdx": "MIT",
        "license_evidence": TILELANG_EVIDENCE,
    },
    {
        "id": "F",
        "name": "zhinianqin/marlin_v100",
        "path": "marlin_v100",
        "requested_ref": "3f16d442cdb4c24dd225bbec196c982a54d9a31c",
        "expected_commit": "3f16d442cdb4c24dd225bbec196c982a54d9a31c",
        "match": "head",
        "role": "reference",
        "license_files": [],
        "spdx": "Apache-2.0",
        "license_evidence": MARLIN_EVIDENCE,
    },
]


def git(repo: Path, *args: str) -> str:
    """Run one git command and return its trimmed stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    """Return True when `older` is reachable from `newer`."""
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def check_head(repo: Path, source: dict, resolved: str) -> list[str]:
    """Apply the head rule for one source and return any errors."""
    expected = source["expected_commit"]
    if source["match"] == "ancestor":
        if not is_ancestor(repo, expected, resolved):
            return [f"base_not_ancestor: {expected} not reachable from {resolved}"]
        return []
    if resolved != expected:
        return [f"head_mismatch: expected {expected}, found {resolved}"]
    return []


def license_blob_digest(repo: Path, commit: str, name: str) -> str | None:
    """Hash one license file at the locked commit.

    Reading from the commit, not from disk, keeps the digest stable when the
    product-base worktree carries campaign edits. The audit re-checks the
    same blob, so both tools must use the same source.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{name}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def check_licenses(repo: Path, source: dict, commit: str) -> tuple[list[dict], list[str]]:
    """Collect license digests and return any errors.

    A source without a LICENSE file must still carry a resolved SPDX
    identifier and written evidence. The audit fails closed otherwise.
    """
    entries: list[dict] = []
    errors: list[str] = []

    for name in source["license_files"]:
        digest = license_blob_digest(repo, commit, name)
        if digest is None:
            errors.append(f"missing_license_file: {name}")
        entries.append({"file": name, "sha256": digest})

    if not source["license_files"]:
        entries.append({"file": None, "sha256": None, "note": "no-license-file"})

    if not source.get("spdx"):
        errors.append("unresolved_license: no SPDX identifier")
    if not source.get("license_evidence"):
        errors.append("unresolved_license: no written evidence")

    return entries, errors


def collect(source: dict) -> dict:
    """Collect every machine fact for one source."""
    repo = WORKSPACE / source["path"]

    if not (repo / ".git").exists():
        return {
            "id": source["id"],
            "name": source["name"],
            "role": source["role"],
            "local_path": str(repo),
            "errors": ["missing_checkout"],
        }

    head = git(repo, "rev-parse", "HEAD")
    errors = check_head(repo, source, head)

    # The product base is pinned to its expected commit for the whole
    # campaign. Recording HEAD here would let a permitted re-lock silently
    # advance the base past the tag, because the ancestor rule accepts any
    # descendant. The current head is recorded separately as an observation.
    if source["match"] == "ancestor":
        resolved = source["expected_commit"]
    else:
        resolved = head

    dirty_lines = [line for line in git(repo, "status", "--porcelain").splitlines() if line]

    # A donor or reference checkout must stay clean for the whole campaign.
    # The product base is the working repository, so campaign edits make it
    # dirty by construction. Record its dirty state as a fact instead.
    if dirty_lines and source["role"] != "product_base":
        errors.append(f"dirty_worktree: {len(dirty_lines)} entries")

    licenses, license_errors = check_licenses(repo, source, resolved)
    errors.extend(license_errors)

    return {
        "id": source["id"],
        "name": source["name"],
        "role": source["role"],
        "repository_url": git(repo, "config", "--get", "remote.origin.url"),
        "requested_ref": source["requested_ref"],
        "expected_commit": source["expected_commit"],
        "match_rule": source["match"],
        "resolved_commit": resolved,
        "observed_head": head,
        "commit_date": git(repo, "log", "-1", "--format=%cI", resolved),
        "commit_subject": git(repo, "log", "-1", "--format=%s", resolved),
        "tree_sha": git(repo, "rev-parse", f"{resolved}^{{tree}}"),
        "spdx": source.get("spdx"),
        "license_evidence": source.get("license_evidence"),
        "license_files": licenses,
        "local_path": str(repo),
        "dirty_entry_count": len(dirty_lines),
        "observed_remote_head": (
            git(repo, "rev-parse", "origin/HEAD")
            or git(repo, "rev-parse", "origin/main")
            or None
        ),
        "errors": errors,
    }


def check_campaign_branch(repo: Path) -> tuple[str, list[str]]:
    """Verify that the product base sits on the campaign branch.

    Returns the observed branch and any errors. A detached head is recorded
    explicitly, because a detached descendant would otherwise pass silently.
    """
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        return "DETACHED", [
            f"detached_head: expected branch {CAMPAIGN_BRANCH}"
        ]
    if branch != CAMPAIGN_BRANCH:
        return branch, [
            f"wrong_branch: expected {CAMPAIGN_BRANCH}, found {branch}"
        ]
    return branch, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--relock",
        action="store_true",
        help=(
            "Replace an existing lock. Allowed only at a phase boundary, "
            "with operator approval."
        ),
    )
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.relock:
        print(f"refusing to overwrite existing lock {out}", file=sys.stderr)
        print("the lock is immutable after approval", file=sys.stderr)
        print("pass --relock only at an approved phase boundary", file=sys.stderr)
        return 1

    sources = [collect(source) for source in SOURCES]
    failed = [s for s in sources if s.get("errors")]

    base_repo = WORKSPACE / "lmdeploy"
    branch, branch_errors = check_campaign_branch(base_repo)
    if branch_errors:
        for entry in sources:
            if entry["id"] == "A":
                entry.setdefault("errors", []).extend(branch_errors)
        failed = [s for s in sources if s.get("errors")]

    lock = {
        "schema_version": 1,
        "campaign": "lmdeploy-v100-next",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORKSPACE),
        "product_base": {
            "repository": "InternLM/lmdeploy",
            "ref": "v0.16.0",
            "commit": "1208bf006bbac69f1f012ceafeeeb70f623b632c",
            "required_branch": CAMPAIGN_BRANCH,
            "observed_branch": branch,
            "observed_head": git(base_repo, "rev-parse", "HEAD"),
        },
        "sources": sources,
        "status": "FAIL" if failed else "OK",
    }

    # A failed run must never replace a good lock. Write diagnostics aside.
    target = out if not failed else out.with_suffix(".failed.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    print(f"wrote {target}")
    print(f"status: {lock['status']}")
    for source in failed:
        print(f"  {source['id']} {source['name']}: {source['errors']}")
    if failed:
        print("lock not written: diagnostics only", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
