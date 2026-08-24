# Approval record

This file records every operator approval for the campaign.

Each entry records the date, the approved scope, and the artifact digest.

No entry may be edited after it is recorded.

## Pending approval: Phase 0A and Phase 1

Status: AWAITING OPERATOR APPROVAL.

The master specification requires a stop here.

### Scope presented for approval

| Item | Value |
| --- | --- |
| Product base | `v0.16.0`, commit `1208bf006bbac69f1f012ceafeeeb70f623b632c` |
| Branch | `experiment/lmdeploy-v100-next` |
| Source lock SHA-256 | `0cc1230a863b60f9dea4ebb64ae2febdc620ee223d9efb1f3ca3b9a56cbe4dfc` |
| Inventory SHA-256 | `bb7b0e11b473bd6400062dbe4754abc54cb19be8202b71fa6a8480e6829126ca` |
| Inventory records | 641 |

Regenerate both digests with the commands in `provenance.md`.

### Artifacts presented

```text
docs/v100/source-lock.json
docs/v100/provenance.md
docs/v100/patch-matrix.md
docs/v100/donor-inventory-summary.json
benchmarks/v100/results/donor-inventory.jsonl
tools/v100/make_source_lock.py
tools/v100/audit_turbomind_deltas.py
```

### Decisions requested

1. Approve the product base and the source lock as immutable.
2. Approve the license resolutions for sources E and F.
3. Acknowledge the corrected SM70 decode finding, recorded below.
4. Approve the Phase 2 gate as a campaign-level gate.
5. Name the small FP16 baseline model, which the specification still leaves open.
6. Approve GPU time on the free island for Phase 0B.

### Correction: the SM70 decode finding was wrong

Recorded at the end of Phase 0A. This entry is append-only.

An earlier finding claimed that donor `MMA_884_DEC` could not host a
block-scaled reader, and that the campaign outcome therefore rode on a route
the donor reported as faulty.

That claim was wrong about the product base.

The base contains no `MMA_884_DEC` symbol and no `decoding_config.h`. It uses
a runtime registry, and `kernel/decoding_sm70_256.cu` already registers SM70
head_dim 256 decode for FP16, INT8, and INT4 KV. Qwen3.8 selects
`KT<half, uint8_t, 3>`, which exists today.

The donor constraint is real but donor-only.

The Phase 2 gate remains mandatory. Registration is not proof of correct
execution.

See `capability-preflight.md` for the evidence.

An independent reviewer confirmed this correction and added two limits, which
the specification now records:

1. The SM70 FP8 GEMM result does not prove FP8 KV attention support.
2. Block-64 scaling needs a new parameter fetch and layout, not only a new
   converter.

### Immutability

The lock is immutable after approval.

`make_source_lock.py` refuses to overwrite an existing lock.

A replacement requires the explicit `--relock` flag.

Use that flag only at a phase boundary, with recorded approval.

A failed run writes `source-lock.failed.json` and never touches the lock.

### Blocking question for the operator

The Phase 1 audit found that quantized KV at head_dim 256 must decode on the
SIMT route, and that the donor remedy for the reported SIMT fault cannot serve
quantized KV.

Phase 2 must close that gate before Phase 4 selects a kernel.

Phase 2 needs GPU access on `gpu-01`.

Confirm the free NVLink island before Phase 2 starts.

### Independent review

An independent reviewer ran four adversarial rounds against these artifacts.

| Round | Result | Findings |
| --- | --- | --- |
| 1 | NO-GO | 4 P1, 2 P2 |
| 2 | NO-GO | 3 P1, 2 P2 |
| 3 | NO-GO | 2 P1, 3 P2 |
| 4 | GO | 3 P2, 1 P3, all fixed after the round |

Every finding from all four rounds is fixed and verified.

The two most serious were provenance defects that would have corrupted the
audit silently:

1. The audit read live checkouts, so a later edit could change recorded
   digests. Content now comes from the locked commit.
2. The lock recorded the current head for the product base, so a permitted
   re-lock would have advanced the comparison base past `v0.16.0`. The lock
   now pins the base to its expected commit and records the head separately.

## Approval log

| Date | Scope | Artifact digest | Approver | Result |
| --- | --- | --- | --- | --- |
| pending | Phase 0A and Phase 1 | see above | pending | pending |
