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
| `PHASE0B-FP8-PACK` | C, F | 0B | IMPLEMENTED, PENDING_REVIEW |

## Implemented family: PHASE0B-FP8-PACK

This family is implemented and measured. It still needs reviewer approval.

The technique comes from donors C and F. No donor code is copied, so this is a
re-implementation against the product base, not a port.

| Field | Value |
|---|---|
| Patch-family ID | `PHASE0B-FP8-PACK` |
| Commit SHA | `c46dbf06`, `f7b18471` |
| Donor file and blob hash | Technique only. See `donor-inventory.jsonl` for `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` in donor C and the `fp8` scale paths in donor F. |
| Donor symbol | `fp8_sm70_prepare` in donor C. `marlin_utils_fp8` scale preparation in donor F. |
| Current upstream equivalent | `src/turbomind/core/data_format.cc`, `lmdeploy/turbomind/weight_format.py`, `lmdeploy/turbomind/converter.py` |
| Dependencies | Existing SM70 `Config_E4M3` tiles. No new kernel. |
| SPDX license identifier | Apache-2.0 for C and F. Both resolve. |
| Notice requirements | None. No donor source is copied. |
| Classification | `V100_CORRECTNESS` |
| Compatibility decision | Pre-SM90 only. Capability 9.0 and above keep the `{128, 128}` form, FP32 scales and the original dequantization. |
| Test evidence | Phase 0B gate on island 2: 8 of 8 semantic checks, long-context needle retrieval PASS, degenerate-output detector False. |
| Service evidence | Qwen3.8-27B-FP8 at TP4, 208.1 tok/s, 23.35 GiB per GPU, 30.4 s load, clean shutdown. |
| Port decision | PENDING_REVIEW |
| Reviewer approval | PENDING |

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
