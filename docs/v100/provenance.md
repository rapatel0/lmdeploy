# Provenance report

This report records machine facts from Phase 0A and Phase 1.

It records no port decision. Port decisions belong in `patch-matrix.md`.

## Product base

| Item | Value |
| --- | --- |
| Repository | InternLM/lmdeploy |
| Ref | `v0.16.0` |
| Commit | `1208bf006bbac69f1f012ceafeeeb70f623b632c` |
| Branch | `experiment/lmdeploy-v100-next` |
| License | Apache-2.0, LICENSE file at root |

The local checkout was 73 commits behind this tag before Phase 0A.

Phase 0A fetched the `upstream` remote and branched from the tag.

The `origin` remote is a fork and does not carry the tag by itself.

## Source lock

The lock file is `docs/v100/source-lock.json`.

Regenerate it with `tools/v100/make_source_lock.py`.

The tool fails closed on a missing checkout, a moved head, a dirty donor
worktree, a wrong campaign branch, a detached head, a missing license file,
or an unresolved license.

The product base is exempt from the dirty check only, because campaign edits
live in that repository. Its content still enters the audit at the locked
commit, so neither working-tree edits nor campaign commits can change any
recorded digest.

The lock is immutable. The tool refuses to overwrite an existing lock, and a
replacement needs the explicit `--relock` flag at an approved phase boundary.
A failed run writes `source-lock.failed.json` and leaves the lock untouched.

## Licenses

| ID | Source | SPDX | Evidence |
| --- | --- | --- | --- |
| A | lmdeploy | Apache-2.0 | LICENSE file at root |
| B | zh-nj/lmdeploy-v100 | Apache-2.0 | LICENSE file at root |
| C | 1CatAI/1Cat-vLLM | Apache-2.0 | LICENSE file at root |
| D | haohervchb/sglang-V100 | Apache-2.0 | LICENSE file at root |
| E | haohervchb/Tilelang-FA-V100 | MIT | README.md License section. No LICENSE file at root. |
| F | zhinianqin/marlin_v100 | Apache-2.0 | Per-file headers in `csrc/`. No LICENSE file at root. |

Sources E and F carry no LICENSE file at their roots.

The master specification requires a resolved license before Phase 1.

Both resolved, so both stayed in the campaign.

Source F carries the Marlin and Neural Magic Apache-2.0 notice in
`csrc/quantization/marlin/marlin.cu`. Preserve that notice on any port.

## Donor inventory

The inventory file is `benchmarks/v100/results/donor-inventory.jsonl`.

The summary file is `docs/v100/donor-inventory-summary.json`.

Regenerate both with `tools/v100/audit_turbomind_deltas.py`.

The audit reads every file at its locked commit, never from the working tree
and never from the branch head. Each record carries
`content_source: locked_git_commit`.

The product base is indexed at its locked commit, so campaign commits cannot
move the comparison base. The summary records the locked commit, the current
head, and the current branch as separate fields.

The audit revalidates every checkout against the lock before any read. It
fails on commit drift, tree drift, dirty donor state, or license digest drift.
The product base uses the ancestor rule; donors must match exactly.

Each record also carries `symbols`, `dependencies`, and `revisions` facts,
which give the patch matrix a starting commit-to-symbol map.

The symbol scan is lexical, not a parse. Every record states its method in
`symbol_extraction` and its cap state in `symbols_truncated`. The dependency
and revision facts carry the same truncation flags.

A reviewer must confirm every symbol before a port.

An empty symbol list is not proof that a file defines no symbol.

Symbol coverage on the current run is 615 of 641 records. The six source
files with no symbol are wrapper headers, package initialisers, and a
binding file that genuinely declare none.

### Delta states

These states are content facts, not campaign classifications:

| State | Meaning |
| --- | --- |
| `identical_content` | Byte-identical to some product-base file |
| `modified` | A product-base file shares the path, content differs |
| `donor_only` | No product-base counterpart |

Comparison uses content digests, not paths. Donor C embeds a partial copy of
an older LMDeploy tree at shared product paths, so a path comparison alone
would misreport it.

### Counts

Total records: 641. Product base indexed 1139 source digests and 22 build
digests across 1187 tracked paths.

Source and build files are compared against separate digest indexes, so a
build file is never matched against a source file.

| ID | Source | Files | Source deltas | Build deltas |
| --- | --- | --- | --- | --- |
| B | zh-nj/lmdeploy-v100 | 445 | 142 identical, 181 modified, 104 donor-only | 16 modified, 2 donor-only |
| C | 1CatAI/1Cat-vLLM | 139 | 76 identical, 57 modified, 6 donor-only | none |
| D | haohervchb/sglang-V100 | 10 | 10 donor-only | none |
| E | haohervchb/Tilelang-FA-V100 | 9 | 9 donor-only | none |
| F | zhinianqin/marlin_v100 | 38 | 36 donor-only | none |

### Reading the counts

The delta columns count compared files only.

A compared file is a source file or a build file. Every other tracked file
reads as `not_compared`.

Source F also tracks two `.gitignore` files, which fall in the `other`
category and carry no delta.

Source B shares 142 files with the product base without change. Those files
need no port. Its 181 modified files are the audit surface for Phase 2.

Source C embeds 139 files, of which 76 are byte-identical to upstream. Its
57 modified files differ from a product-base file at the same path, and its
6 unmatched files have no product-base counterpart.

Those states record content facts only. They do not classify behavior.
Behavior classification belongs in `patch-matrix.md`.

Sources D, E, and F share no path with the product base, so every compared
file reads as `donor_only`. That state means "no counterpart", not "new work".

## Phase 1 finding: SM70 decode routing

The audit surfaced a campaign-level risk. Full detail sits in the master
specification, under the Phase 2 SM70 decode correctness gate.

Summary of the machine facts:

1. Donor commit `c1e21e24` reports a bisect blaming the SIMT decode route for
   garbage output on Qwen3.5-122B at TP8 with MoE. Its remedy enables
   `MMA_884_DEC`.
2. Donor `decoding_config.h` routes head_dim 256 with quantized KV to
   `MMA_SIMT` unconditionally.
3. Donor `impl_884_decode.h` declares `using Tkv = T_`, so `MMA_884_DEC`
   cannot accept a KV type distinct from the activation type.
4. The product base contains no `MMA_884_DEC` symbol.

Consequence: any quantized KV at head_dim 256 must decode on the SIMT route,
and the donor's own remedy cannot serve it.

The donor reproducer is Qwen3.5-122B with MoE. The campaign target is
Qwen3.8-27B with Gated DeltaNet. The report is therefore a risk to test, not
an established fault for this target.

Phase 2 must close this gate before Phase 4 selects a kernel.

## Reproduction

```bash
cd ~/repos/cust-lmdeploy/lmdeploy
# The lock already exists and is immutable. Add --relock only at an approved
# phase boundary.
python3 tools/v100/make_source_lock.py --out docs/v100/source-lock.json
python3 tools/v100/audit_turbomind_deltas.py \
    --lock docs/v100/source-lock.json \
    --out benchmarks/v100/results/donor-inventory.jsonl \
    --summary docs/v100/donor-inventory-summary.json
```

Both commands exit non-zero on any provenance failure.

The summary binds both artifacts. It records the SHA-256 digest of the source
lock and the SHA-256 digest of the inventory file.

| Artifact | SHA-256 |
| --- | --- |
| `docs/v100/source-lock.json` | `0cc1230a863b60f9dea4ebb64ae2febdc620ee223d9efb1f3ca3b9a56cbe4dfc` |
| `benchmarks/v100/results/donor-inventory.jsonl` | `bb7b0e11b473bd6400062dbe4754abc54cb19be8202b71fa6a8480e6829126ca` |

Both tools hash every license file at its locked commit, not from disk, so a
dirty product-base worktree cannot change a recorded license digest.

## Base pinning

The product base entry records `resolved_commit` as the expected tag commit,
never the current head. The current head is recorded separately as
`observed_head`.

That rule keeps a permitted re-lock pinned to `v0.16.0`. Without it, a
re-lock after a campaign commit would advance the comparison base, because
the ancestor rule accepts any descendant.
