# MTP audit

The master specification requires an audit of the old `MTPPredictor` and its
rejection path in Phase 2, and forbids implementing MTP in this campaign.

This is an audit only. Nothing here is ported.

## Provenance

| Fact | Value |
| --- | --- |
| Donor | B, `zh-nj/lmdeploy-v100` |
| Files | `src/turbomind/models/llama/mtp_predictor.h`, `mtp_predictor.cc` |
| Size | 107 and 524 lines |
| Commit | `d7c29f88e440`, one of the five post-root commits the specification names |
| SPDX | Apache-2.0 |
| Delta state | `donor_only` |

The product base contains no `MTPPredictor` symbol. The only `speculative`
match in the base is unrelated scheduler bookkeeping in `engine/scheduler.cc`,
not a draft-token path.

## What the donor implements

`MTPPredictor` is a draft-model forward path:

- `SetupAttention` prepares the attention environment for a phase.
- `LookupEmbedding` gathers input embeddings for a batch of token ids, with a
  tensor-parallel all-gather when `tp_size_` exceeds one.
- `PostEmbedding` projects features to logits, sharded by TP rank.

## There is no rejection path

The specification asks for an audit of the rejection path. The donor does not
contain one. A search for `reject` across `mtp_predictor.cc` returns nothing.

The donor implements draft-token generation only. Acceptance and rejection
logic, which is the part that decides correctness of a speculative decoder,
lives outside these files or does not exist in the donor. A future MTP campaign
must therefore treat the donor as a partial starting point, not as a complete
speculative decoder.

## Blocking finding: the donor carries the global profiler

The specification forbids restoring the old global profiler and requires Nsight
Systems, CUPTI, or scoped runtime telemetry instead.

`mtp_predictor.cc` embeds exactly that profiler:

```cpp
static FILE* g_mtp_profile_fp   = nullptr;
static int   g_mtp_profile_iter = 0;

static void mtp_profile_init()
{
    if (!g_mtp_profile_fp) {
        g_mtp_profile_fp = fopen("/tmp/mtp_profile.csv", "w");
```

Two mutable file-scope globals and a hardcoded path in `/tmp`, written from a
macro on a per-layer, per-step basis. This is process-global mutable state in a
multi-rank process, and the path does not carry a rank suffix, so every rank on
a node writes to one file.

A future MTP campaign must strip this profiler before porting anything.

## Recommendation

Do not port `MTPPredictor` in this campaign, which matches the specification.

Record three constraints for the separate MTP campaign:

1. The donor supplies draft generation only. Acceptance and rejection must be
   designed, not copied.
2. The embedded global profiler must be removed and replaced with scoped
   telemetry.
3. The donor targets a compile-time structure the base has replaced with a
   runtime registry, so a direct port does not apply.
