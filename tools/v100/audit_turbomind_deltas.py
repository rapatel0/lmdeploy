#!/usr/bin/env python3
"""Inventory the donor sources for the lmdeploy-v100-next campaign.

This tool generates machine facts only. It makes no port decision and no
classification. Reviewed classifications belong in docs/v100/patch-matrix.md.

The tool answers one question for every donor file:

    Does this file differ from the locked product base, and how?

Provenance rules:

* Every checkout is revalidated against the source lock before any read.
  The audit fails on commit, tree, dirty-state, or license-digest drift.
* File content comes from the locked commit, not from the working tree and
  not from the branch head. The product base is read at its locked
  `resolved_commit`, so campaign commits cannot change the comparison base.
* Every source ID and every configured scope must be present and non-empty.
* An unreadable object fails the run. Files never disappear silently.

Comparison uses content digests, not paths, because donor C embeds a partial
copy of an older LMDeploy tree that shares product paths.

Outputs JSONL, one record per donor file, plus a JSON summary.

Usage:
    python3 tools/v100/audit_turbomind_deltas.py \\
        --lock docs/v100/source-lock.json \\
        --out benchmarks/v100/results/donor-inventory.jsonl \\
        --summary docs/v100/donor-inventory-summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Caps keep one pathological file from dominating an inventory record.
MAX_SYMBOLS_PER_FILE = 200
MAX_DEPS_PER_FILE = 100
MAX_REVISIONS_PER_FILE = 50

# Lexical symbol scans. These produce candidate names for reviewer
# confirmation. They are not a parse and carry no semantic guarantee.
SYMBOL_PATTERNS_C = [
    re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(?:struct|class|union)\s+(\w+)", re.M),
    re.compile(r"^\s*#define\s+(\w+)", re.M),
    re.compile(r"^\s*using\s+(\w+)\s*=", re.M),
    re.compile(r"^\s*typedef\s+.+?\b(\w+)\s*;", re.M),
    re.compile(r"^\s*enum(?:\s+class)?\s+(\w+)", re.M),
    re.compile(r"^\s*namespace\s+(\w+)", re.M),
    # Function definitions and declarations. This form tolerates any leading
    # run of qualifiers, attributes, and return-type tokens, so CUDA forms
    # such as `inline __device__ float2 name(...)` are matched.
    re.compile(
        r"^[ \t]*(?:(?:template\s*<[^>]*>|inline|static|extern|constexpr|virtual|explicit"
        r"|__global__|__device__|__host__|__forceinline__|__inline__|friend)\s+)*"
        r"(?:[A-Za-z_][\w:<>,\s*&]*?[\s*&])"
        r"([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:const\s*)?(?:noexcept\s*)?[{;]",
        re.M,
    ),
]

# Tokens the function pattern can capture that are never symbol names.
SYMBOL_STOPWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "else",
    "do",
    "catch",
    "throw",
    "case",
    "defined",
    "static_assert",
    "decltype",
    "operator",
    "and",
    "or",
    "not",
    "constexpr",
    "noexcept",
    "alignas",
    "const_cast",
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "delete",
    "new",
    "typeid",
    "assert",
    "printf",
    "break",
    "continue",
}

# Lines that open a control-flow or call context. A function definition never
# starts with one of these, so the function pattern must skip such a line.
# The test anchors at the start of the line, so a definition whose body
# happens to contain `return` on the same line is still accepted.
CONTROL_LINE = re.compile(r"^[ \t]*\}?[ \t]*(?:if|for|while|switch|else|do|catch|return|throw|case)\b")
SYMBOL_PATTERNS_PY = [
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M),
    re.compile(r"^\s*class\s+(\w+)", re.M),
]

DEP_PATTERNS_C = [
    re.compile(r'^\s*#include\s+["<]([^">]+)[">]', re.M),
]
DEP_PATTERNS_PY = [
    re.compile(r"^\s*import\s+([\w.]+)", re.M),
    re.compile(r"^\s*from\s+([\w.]+)\s+import", re.M),
]

# Extensions the audit treats as portable source.
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".pyi",
}

# Build-system files. These are tracked and compared in their own category,
# because a build change can alter kernel registration and build behavior.
BUILD_NAMES = {"CMakeLists.txt", "setup.py", "pyproject.toml", "Makefile"}
BUILD_SUFFIXES = {".cmake", ".mk"}

# Every source ID the audit requires. A missing ID fails the run.
REQUIRED_SOURCE_IDS = ("A", "B", "C", "D", "E", "F")
DONOR_SOURCE_IDS = ("B", "C", "D", "E", "F")

# Donor scopes. Each configured scope must exist and must yield files.
SCOPES = {
    "B": ["src/turbomind"],
    "C": ["csrc/sm70_turbomind"],
    "D": ["python/sglang/srt/layers/attention/tilelang_fa_v100"],
    "E": ["tilelang_fa_v100"],
    "F": ["csrc/quantization/marlin", "csrc/moe"],
}

# Product-base scopes indexed for content comparison.
BASE_INDEX_SCOPES = ["src", "lmdeploy"]

# Donors whose paths map onto product-base paths for a path-wise compare.
# A donor absent from this map has no product-path counterpart.
PATH_MAPPED_DONORS = {"B", "C"}

NOTICE_MARKERS = [
    "OpenMMLab",
    "NVIDIA",
    "Apache License",
    "SPDX-License-Identifier",
    "Copyright",
    "TileLang",
    "Marlin",
    "Neural Magic",
    "vLLM",
    "SGLang",
]


class AuditError(RuntimeError):
    """Raised when the audit cannot proceed on trustworthy inputs."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    """Run one git command and return trimmed stdout, or an empty string."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_bytes(repo: Path, *args: str) -> bytes:
    """Run one git command and return raw stdout.

    Raises AuditError when the command fails, so a missing object can never
    silently drop a file from the inventory.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise AuditError(f"git failed in {repo}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result.stdout


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    """Return True when `older` is reachable from `newer`."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def revalidate(entry: dict) -> None:
    """Re-verify one locked checkout before any content read.

    The source lock records a past state. This function proves that the
    checkout can still supply that state. It fails closed on any drift.

    The product base uses the ancestor rule, because the campaign branch adds
    commits on top of the locked base. Its content is still read at the locked
    commit, so later campaign commits cannot move the comparison base. Donor
    and reference checkouts must match their locked commit exactly.
    """
    source_id = entry.get("id", "?")
    repo = Path(entry.get("local_path", ""))
    locked = entry.get("resolved_commit")
    is_base = entry.get("role") == "product_base"

    if not isinstance(locked, str) or not locked:
        raise AuditError(f"source {source_id}: lock has no resolved commit")

    if not (repo / ".git").exists():
        raise AuditError(f"source {source_id}: missing checkout at {repo}")

    # The locked commit must still exist in the checkout, whatever the head is.
    if git_text(repo, "cat-file", "-t", f"{locked}^{{commit}}") != "commit":
        raise AuditError(f"source {source_id}: locked commit {locked} is not present")

    head = git_text(repo, "rev-parse", "HEAD")
    if is_base:
        if not is_ancestor(repo, locked, head):
            raise AuditError(f"source {source_id}: locked base {locked} is not an ancestor of {head}")
    elif head != locked:
        raise AuditError(f"source {source_id}: commit drift, lock has {locked}, checkout has {head}")

    tree = git_text(repo, "rev-parse", f"{locked}^{{tree}}")
    if tree != entry.get("tree_sha"):
        raise AuditError(
            f"source {source_id}: tree drift at {locked}, lock has {entry.get('tree_sha')}, checkout has {tree}"
        )

    # Campaign edits live in the product base, so it is dirty by design.
    # Donor and reference checkouts must remain clean.
    #
    # Use the strict helper. A lenient helper would turn a failed status
    # query into an empty string, which would read as a clean checkout.
    if not is_base:
        raw = git_bytes(repo, "status", "--porcelain")
        dirty = [line for line in raw.decode("utf-8", errors="replace").splitlines() if line]
        if dirty:
            raise AuditError(f"source {source_id}: dirty worktree, {len(dirty)} entries")

    for record in entry.get("license_files", []):
        name = record.get("file")
        if name is None:
            continue
        # Compare the license blob at the locked commit, not on disk.
        try:
            data = git_bytes(repo, "show", f"{locked}:{name}")
        except AuditError:
            raise AuditError(f"source {source_id}: license file {name} is absent at {locked}") from None
        if sha256_bytes(data) != record.get("sha256"):
            raise AuditError(f"source {source_id}: license digest drift for {name}")


def list_tree(repo: Path, commit: str, scope: str) -> list[str]:
    """List every tracked file path under one scope of the locked commit."""
    args = ["ls-tree", "-r", "--name-only", "-z", commit]
    if scope:
        args.append(scope)
    raw = git_bytes(repo, *args)
    return [p for p in raw.decode("utf-8", errors="replace").split("\0") if p]


def read_locked(repo: Path, commit: str, rel: str) -> bytes:
    """Read one file's content at the locked commit."""
    return git_bytes(repo, "show", f"{commit}:{rel}")


def list_symbols(rel: str, text: str) -> tuple[list[str], bool]:
    """Extract candidate symbol names from one source file.

    Returns the names and a truncation flag.

    This is a lexical scan, not a parse. It produces candidates for reviewer
    confirmation. A reviewer must confirm every symbol before a port, and
    must not treat an empty list as proof that a file defines no symbol.
    """
    suffix = Path(rel).suffix
    names: list[str] = []
    if suffix in {".py", ".pyi"}:
        patterns = SYMBOL_PATTERNS_PY
    elif suffix in SOURCE_SUFFIXES:
        patterns = SYMBOL_PATTERNS_C
    else:
        return [], False
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = match.group(1)
            if not name or name in SYMBOL_STOPWORDS or name in names:
                continue
            # Reject a match whose own line opens a control-flow statement or
            # an assignment, because those are call sites, not definitions.
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.start())
            line = text[line_start : line_end if line_end != -1 else len(text)]
            if CONTROL_LINE.match(line):
                continue
            names.append(name)
    truncated = len(names) > MAX_SYMBOLS_PER_FILE
    return names[:MAX_SYMBOLS_PER_FILE], truncated


def list_dependencies(rel: str, text: str) -> list[str]:
    """Extract include and import targets from one source file."""
    suffix = Path(rel).suffix
    deps: list[str] = []
    if suffix in {".py", ".pyi"}:
        patterns = DEP_PATTERNS_PY
    elif suffix in SOURCE_SUFFIXES:
        patterns = DEP_PATTERNS_C
    else:
        return []
    for pattern in patterns:
        for match in pattern.finditer(text):
            dep = match.group(1)
            if dep and dep not in deps:
                deps.append(dep)
    return deps[:MAX_DEPS_PER_FILE]


def file_revisions(repo: Path, commit: str, rel: str) -> tuple[list[str], bool]:
    """Return the commits that touched one path, up to the locked commit.

    Uses the strict Git helper, so a failed history query raises instead of
    recording an empty list. Returns the commits and a truncation flag.
    """
    raw = git_bytes(repo, "rev-list", commit, "--", rel)
    revisions = raw.decode("utf-8", errors="replace").split()
    truncated = len(revisions) > MAX_REVISIONS_PER_FILE
    return revisions[:MAX_REVISIONS_PER_FILE], truncated


def category_for(rel: str) -> str:
    """Classify one path as source, build, or other.

    Build names are checked first. `setup.py` carries a source suffix but is
    a build file, so a suffix-first order would misclassify it.
    """
    path = Path(rel)
    if path.name in BUILD_NAMES or path.suffix in BUILD_SUFFIXES:
        return "build"
    if path.suffix in SOURCE_SUFFIXES:
        return "source"
    return "other"


def find_notices(text: str) -> list[str]:
    """Return the notice markers present near the top of a file."""
    head = text[:4000]
    return [marker for marker in NOTICE_MARKERS if marker in head]


def build_base_index(repo: Path, commit: str) -> tuple[dict[str, dict[str, str]], set[str]]:
    """Index the locked product base by content digest, per category.

    Returns a category-keyed digest map and the set of tracked paths. Build
    files are indexed separately from source files, so a build comparison
    never consults the source index.
    """
    by_digest: dict[str, dict[str, str]] = {"source": {}, "build": {}}
    paths: set[str] = set()
    for scope in BASE_INDEX_SCOPES:
        for rel in list_tree(repo, commit, scope):
            paths.add(rel)
            category = category_for(rel)
            if category not in by_digest:
                continue
            data = read_locked(repo, commit, rel)
            by_digest[category].setdefault(sha256_bytes(data), rel)
    return by_digest, paths


def base_counterpart(source_id: str, rel: str) -> str | None:
    """Map one donor path onto its product-base path, when one exists."""
    if source_id == "B":
        return rel
    if source_id == "C":
        parts = Path(rel).parts
        if "lmdeploy" in parts:
            tail = parts[parts.index("lmdeploy") + 1 :]
            if tail:
                return str(Path(*tail))
    return None


def classify_delta(
    digest: str,
    index: dict[str, str],
    base_path: str | None,
    base_paths: set[str],
) -> tuple[str, str | None]:
    """Return the machine delta state for one donor file.

    `index` must be the digest map for the file's own category.

    States are content facts, not campaign classifications:
        identical_content -- byte-identical to some product-base file
        modified          -- a product-base file shares the path, content differs
        donor_only        -- no product-base counterpart
    """
    if digest in index:
        return "identical_content", index[digest]
    if base_path is not None and base_path in base_paths:
        return "modified", None
    return "donor_only", None


def audit_source(
    entry: dict,
    by_digest: dict[str, dict[str, str]],
    base_paths: set[str],
) -> list[dict]:
    """Produce one record for every tracked file in one donor scope."""
    source_id = entry["id"]
    repo = Path(entry["local_path"])
    commit = entry["resolved_commit"]
    scopes = SCOPES[source_id]

    records: list[dict] = []
    for scope in scopes:
        listing = list_tree(repo, commit, scope)
        if not listing:
            raise AuditError(f"source {source_id}: scope '{scope}' is absent or empty at {commit}")
        for rel in sorted(listing):
            data = read_locked(repo, commit, rel)
            digest = sha256_bytes(data)
            category = category_for(rel)

            base_path = base_counterpart(source_id, rel) if source_id in PATH_MAPPED_DONORS else None

            delta, match = "not_compared", None
            if category in ("source", "build"):
                # Compare within the file's own category, so a build file is
                # never matched against the source digest index.
                delta, match = classify_delta(digest, by_digest[category], base_path, base_paths)

            text = data.decode("utf-8", errors="replace")
            symbols, symbols_truncated = list_symbols(rel, text)
            revisions, revisions_truncated = file_revisions(repo, commit, rel)
            dependencies = list_dependencies(rel, text)

            records.append(
                {
                    "source_id": source_id,
                    "source_name": entry["name"],
                    "source_commit": commit,
                    "source_tree_sha": entry["tree_sha"],
                    "source_spdx": entry["spdx"],
                    "scope": scope,
                    "donor_path": rel,
                    "blob_sha256": digest,
                    "size_bytes": len(data),
                    "line_count": text.count("\n"),
                    "category": category,
                    "delta": delta,
                    "base_match_path": match,
                    "base_path_checked": base_path,
                    "symbols": symbols,
                    "symbol_extraction": "lexical_scan",
                    "symbols_truncated": symbols_truncated,
                    "dependencies": dependencies,
                    "dependencies_truncated": len(dependencies) >= MAX_DEPS_PER_FILE,
                    "revisions": revisions,
                    "revisions_truncated": revisions_truncated,
                    "notices": find_notices(text),
                    "content_source": "locked_git_commit",
                }
            )
    return records


def load_lock(path: Path) -> tuple[dict, bytes]:
    """Load and structurally validate the source lock."""
    try:
        raw = path.read_bytes()
        lock = json.loads(raw)
    except OSError as exc:
        raise AuditError(f"cannot read source lock {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"source lock {path} is not valid JSON: {exc}") from exc

    if not isinstance(lock, dict):
        raise AuditError("source lock root is not an object")
    if lock.get("status") != "OK":
        raise AuditError(f"source lock status is {lock.get('status')}, refusing to audit")

    sources = lock.get("sources")
    if not isinstance(sources, list):
        raise AuditError("source lock has no sources list")

    by_id: dict[str, dict] = {}
    for entry in sources:
        if not isinstance(entry, dict) or "id" not in entry:
            raise AuditError("source lock contains a malformed source entry")
        by_id[entry["id"]] = entry

    missing = [sid for sid in REQUIRED_SOURCE_IDS if sid not in by_id]
    if missing:
        raise AuditError(f"source lock is missing required sources: {missing}")

    for sid, entry in by_id.items():
        for field in ("name", "local_path", "resolved_commit", "tree_sha", "spdx"):
            if not entry.get(field):
                raise AuditError(f"source {sid} is missing required field '{field}'")

    try:
        lock["product_base"]["commit"]
    except (KeyError, TypeError) as exc:
        raise AuditError(f"source lock has no product base commit: {exc}") from exc

    return lock, raw


def summarize(records: list[dict], source_id: str) -> dict:
    """Build the per-source summary block."""
    subset = [r for r in records if r["source_id"] == source_id]
    compared = [r for r in subset if r["delta"] != "not_compared"]
    return {
        "name": subset[0]["source_name"],
        "commit": subset[0]["source_commit"],
        "tree_sha": subset[0]["source_tree_sha"],
        "spdx": subset[0]["source_spdx"],
        "file_count": len(subset),
        "category_counts": dict(Counter(r["category"] for r in subset)),
        "delta_counts_source": dict(Counter(r["delta"] for r in compared if r["category"] == "source")),
        "delta_counts_build": dict(Counter(r["delta"] for r in compared if r["category"] == "build")),
        "files_without_notice": sum(1 for r in compared if r["category"] == "source" and not r["notices"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, help="Path to source-lock.json.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--summary", required=True, help="Output summary JSON path.")
    args = parser.parse_args()

    try:
        lock, lock_bytes = load_lock(Path(args.lock))
        by_id = {s["id"]: s for s in lock["sources"]}

        print("revalidating locked checkouts ...")
        for sid in REQUIRED_SOURCE_IDS:
            revalidate(by_id[sid])
            print(f"  {sid} ok")

        base_repo = Path(by_id["A"]["local_path"])
        base_commit = by_id["A"]["resolved_commit"]
        print(f"indexing product base at locked commit {base_commit[:12]} ...")
        by_digest, base_paths = build_base_index(base_repo, base_commit)
        print(
            f"  indexed {len(by_digest['source'])} source and "
            f"{len(by_digest['build'])} build digests, "
            f"{len(base_paths)} tracked paths"
        )

        records: list[dict] = []
        for sid in DONOR_SOURCE_IDS:
            entry = by_id[sid]
            print(f"auditing {sid} {entry['name']} ...")
            found = audit_source(entry, by_digest, base_paths)
            if not found:
                raise AuditError(f"source {sid}: no records produced")
            print(f"  {len(found)} files")
            records.extend(found)
    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    out.write_text(payload)

    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_lock_path": str(Path(args.lock)),
        "source_lock_sha256": sha256_bytes(lock_bytes),
        "inventory_path": str(out),
        "inventory_sha256": sha256_bytes(payload.encode("utf-8")),
        "inventory_record_count": len(records),
        "product_base_commit": lock["product_base"]["commit"],
        "product_base_locked_commit": base_commit,
        "product_base_tree_sha": by_id["A"]["tree_sha"],
        "product_base_current_head": git_text(base_repo, "rev-parse", "HEAD"),
        "product_base_current_branch": git_text(base_repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "base_indexed_source_digests": len(by_digest["source"]),
        "base_indexed_build_digests": len(by_digest["build"]),
        "base_tracked_paths": len(base_paths),
        "content_source": "locked_git_commit",
        "per_source": {sid: summarize(records, sid) for sid in DONOR_SOURCE_IDS},
        "note": (
            "Machine facts only. Delta states are content comparisons, not "
            "campaign classifications. Reviewed classifications and port "
            "decisions belong in docs/v100/patch-matrix.md."
        ),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {out}")
    print(f"wrote {summary_path}")
    for sid, stats in summary["per_source"].items():
        print(f"  {sid} {stats['name']}: source={stats['delta_counts_source']} build={stats['delta_counts_build']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
