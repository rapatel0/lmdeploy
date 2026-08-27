# TurboMind MTP qualification — partial acceptance implemented, still rejected

Date: 2026-08-28

Hardware: 4x Tesla V100-SXM2-32GB, TP4, FP16 execution of Qwen3.8-27B-FP8

Model: `/models/Qwen3.8-27B-FP8`

## Decision

Do not enable TurboMind MTP by default. Keep `num_draft_tokens=0`.

TurboMind now retains accepted draft prefixes for the recurrent Gated DeltaNet
(GDN) model. This removes the previous requirement for full K-token acceptance
and more than doubles K=4 throughput. It still does not beat target-only decode:

| Depth | Committed tokens / verification forward | Decode | Ratio vs K=0 | Decision |
| ---: | ---: | ---: | ---: | --- |
| K=0 | 1.00 / 1 | 55.02 tok/s | 1.000x | baseline |
| K=1 | 1.31 / 2 | 42.62 tok/s | 0.775x | reject |
| K=4 | 1.40 / 5 | 35.62 tok/s | 0.647x | reject |

The remaining limitation is draft quality and draft/verification cost, not an
all-or-nothing acceptance policy or idle tensor-parallel ranks.

## Partial-prefix recurrent-state commit

The V100 path now preserves the GDN frontier after each verification input and
selects the frontier corresponding to the committed prefix:

- fused Conv1d captures each intermediate convolution-state frontier;
- the pre-SM90 chunked delta-rule kernel captures each intermediate recurrent
  state using the same arithmetic and tile geometry as ordinary recurrent
  decode;
- verification records slot zero as the pre-forward state and slots `1..K+1`
  after successive verifier inputs;
- accepting `n` drafts plus the verifier bonus selects slot `n+1`;
- an accepted EOS draft omits the bonus and selects slot `n`;
- forced rejection commits the position-zero verifier bonus and selects slot 1;
- if the active batch exceeds the fixed snapshot budget, the exact fallback
  restores slot zero, commits nothing, and performs an ordinary decode;
- intermediate capture is reset before every forward and enabled only for the
  current verification forward, preventing a later long prompt prefill from
  writing beyond the speculative slot allocation.

The design reuses the fixed snapshot allocation rather than allocating one full
conv-plus-recurrent snapshot for every possible row and draft position.

Commit: `d47ccac69311`

## Correctness evidence

### Force-rejection harness

`TM_MTP_FORCE_REJECT` now injects invalid draft IDs before `GreedyReject` rather
than clearing the accepted count afterward. The old harness retained a bonus
from the originally accepted prefix and skipped `target[0]`; the corrected
harness rejects naturally at position zero and publishes the right bonus.

With partial state selection, both normal and forced K=4 complete without the
prior crash or five-row divergence. Four of five fresh-process rows are
byte-identical. The only remaining text split is row 2 at output position 3,
the same unstable FP16 near-tie repeatedly observed in K=0-versus-K=0 controls.

Artifact:
`/localpool/lmdeploy-v100-next/results/20260828_023857-identity-d47ccac69311`

### Fresh-process token equality is not a valid oracle at the near-tie

Independent K=0 processes have repeatedly flipped row 2 at output position 3
(`4087` versus `5707`), including with:

- ordinary execution;
- `CUDA_LAUNCH_BLOCKING=1`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The ordinary top-k reduction and speculative verifier now both resolve exact
ties using a value-descending, token-ID-ascending total order. This removes
reduction-tree-dependent exact-tie selection, but it cannot eliminate small
fresh-process FP16 logit differences before argmax.

### Same-process verifier/replay parity

`TM_SPEC_LOGIT_PARITY=1` retained verifier logits and compared them with the
next ordinary replay in the same process after rollback. On the final partial
state implementation:

- comparisons: 20;
- same argmax: 20/20;
- maximum absolute logit error: 0.07421875;
- maximum RMS logit error: 0.0131.

The remaining fresh-process row-2 text split is therefore not a measured
same-process decision mismatch.

Artifact:
`/localpool/lmdeploy-v100-next/results/20260828_014213-spectrace-6dc8d64ff913`

## Performance evidence

### Final partial-prefix benchmark

Commit: `d47ccac69311`

| Depth | Decode | Inclusive | Mean all-GPU utilization | Maximum memory |
| ---: | ---: | ---: | ---: | ---: |
| K=0 | 55.02 tok/s | 50.80 tok/s | 40.6% | 28,462 MiB |
| K=1 | 42.62 tok/s | 39.97 tok/s | 45.3% | 31,408 MiB |
| K=4 | 35.62 tok/s | 33.84 tok/s | 48.6% | 31,408 MiB |

Acceptance telemetry over the measured streams:

- K=1: 672 committed tokens over 512 verification forwards, commit length
  1.31, with 160 full accepts;
- K=4: 718 committed tokens over 512 verification forwards, commit length
  1.40, with zero full accepts.

K=4 now makes useful progress despite never accepting all four drafts. This is
the intended result of partial-prefix state retention.

Artifact:
`/localpool/lmdeploy-v100-next/results/20260828_030949-gpuutil-d47ccac69311`

### Improvement over all-or-nothing rollback

| Depth | Old decode | Partial-prefix decode | Relative improvement |
| ---: | ---: | ---: | ---: |
| K=1 | 29.82 tok/s | 42.62 tok/s | 1.43x |
| K=4 | 16.00 tok/s | 35.62 tok/s | 2.23x |

Partial acceptance fixed the structural full-acceptance problem, but K=1 still
performs about 29% more GPU work than K=0 per wall-clock interval while
producing 22.5% fewer output tokens. K=4 performs still more draft and verifier
work for only 1.40 committed tokens per target verification.

## Why the SGLang DFlash2 result is still different

The SGLang V100 result of 136.6 tok/s versus its 58.2 tok/s target-only
baseline (2.35x) uses a dedicated five-layer DFlash2 drafter, one anchor plus
seven parallel proposals, E5M2 target KV, CUDA graphs for selector and draft,
and overlap-plan-stream execution. It reported a 3.77-token average commit
length on the 1K workload.

TurboMind now has the corresponding selective recurrent-state publication
mechanism, but not the same draft architecture or commit quality. Its measured
K=4 commit length is 1.40, only 37% of DFlash2's 3.77. The current MTP predictor
also performs sequential draft forwards and cannot amortize them at that commit
rate.

Reference:
`sglang-V100/benchmark/qwen38_27b_fp8_dflash2_e5m2_v100_20260821/README.md`

## Conditions for reopening

Partial recurrent-state commit is no longer the blocker. Reopen qualification
only with a change expected to raise useful work per verification, such as:

1. a substantially stronger or parallel drafter that raises average commit
   length toward the measured break-even point;
2. CUDA-graph capture and launch-overhead reduction for the draft and verifier
   paths;
3. selective drafting that disables MTP where predicted prefix acceptance does
   not amortize its cost;
4. overlap of draft/selector work comparable to the DFlash2 execution plan.

Any reopened path must rerun same-process logit parity, mixed-row retirement,
EOS, forced rejection, and matched end-to-end throughput gates. A throughput
ratio at or below 1.0 remains a rejection regardless of acceptance telemetry.
