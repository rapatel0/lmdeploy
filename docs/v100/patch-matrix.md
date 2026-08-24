# Patch matrix

This file holds reviewed classifications and port decisions.

Machine facts live in `provenance.md` and in the inventory artifacts.

Do not copy machine facts into this file. Reference them instead.

## Status

Phase 1 produced the machine inventory. No port decision is approved yet.

Every row below is `PENDING_REVIEW` until an operator approves it.

## Classifications

The master specification defines these values:

```text
UPSTREAM_EQUIVALENT
UPSTREAM_REWRITE
V100_CORRECTNESS
V100_PERFORMANCE
NEW_QUANT_FORMAT
RUNTIME_ADAPTER_ONLY
EXPERIMENTAL_DEFAULT_OFF
OBSOLETE
UNKNOWN_PROVENANCE
INCOMPATIBLE_LICENSE
```

## Required fields for each patch family

Every approved row must record all of these:

- Patch-family ID.
- Commit SHA.
- Donor file and blob hash.
- Donor symbol.
- Current upstream equivalent.
- Dependencies.
- SPDX license identifier.
- Notice requirements.
- Compatibility decision.
- Test evidence.
- Service evidence.
- Port decision.
- Reviewer approval.

## Pre-classified from the Phase 1 inventory

These rows record classifications that the content audit already settles.

They still need reviewer approval before any port.

| Family ID | Scope | Classification | Basis | Status |
| --- | --- | --- | --- | --- |
| `B-IDENTICAL` | 142 source files in donor B | `UPSTREAM_EQUIVALENT` | Byte-identical to the product base | PENDING_REVIEW |
| `C-IDENTICAL` | 76 source files in donor C | `UPSTREAM_EQUIVALENT` | Byte-identical to the product base | PENDING_REVIEW |
| `B-CODEGEN` | Donor B `src/turbomind/kernels/attention/codegen/` | `OBSOLETE` | Removed macro files. The specification forbids porting them. | PENDING_REVIEW |

No other row is pre-classified. Every remaining file needs review.

## Open candidate families

These families are named in the master specification. Each needs a
commit-to-symbol map before any port.

| Family ID | Source | Phase | Status |
| --- | --- | --- | --- |
| `SM70-DECODE-CORRECTNESS` | B | 2 | Blocked by the Phase 2 gate |
| `WEIGHT-CACHE-CLEANUP` | B | 2 | PENDING_REVIEW |
| `DISPATCH-CACHE-SYNC` | C | 3 | PENDING_REVIEW |
| `ACTIVE-EXPERT-SCHED` | C | 3 | PENDING_REVIEW |
| `SMALL-M-TACTIC` | C | 3 | PENDING_REVIEW |
| `AWQ-PREFILL-WORKSPACE` | C | 3 | PENDING_REVIEW |
| `FP8-EXACT-DENSE-PREFILL` | C | 3 | PENDING_REVIEW |
| `GRAPH-WORKSPACE-OWNERSHIP` | C | 3 | PENDING_REVIEW |
| `COMPACT-REDUCTION-FIX` | C | 3 | PENDING_REVIEW |
| `MARLIN-SM70-GROUPED-SCALE` | F | 3 | PENDING_REVIEW |

## Blocked family: SM70-DECODE-CORRECTNESS

The Phase 1 audit found that the donor remedy cannot serve quantized KV.

See the Phase 1 finding in `provenance.md`.

Do not classify or port this family before the Phase 2 gate closes.

A direct `MMA_884_DEC` port is rejected in advance for quantized KV, because
`impl_884_decode.h` fixes `Tkv` to `T`.

## Rejected in advance

The master specification rejects these without further audit:

| Item | Reason |
| --- | --- |
| vLLM private custom-all-reduce coupling | Private runtime coupling |
| Relative includes escaping the TurboMind subtree | Build boundary violation |
| BFLA sparse attention | Out of campaign scope |
| DFlash verifier kernels | Out of campaign scope |
| FP8 LUT bridge as an INT8 substitute | Not an equivalent format |
| Donor `codegen/` macro files | Removed upstream |
