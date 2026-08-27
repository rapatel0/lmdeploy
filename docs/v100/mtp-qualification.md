# TurboMind MTP qualification — rejected

Date: 2026-08-27  
Hardware: 4x Tesla V100-SXM2-32GB, TP4, FP16 execution of Qwen3.8-27B-FP8  
Model: `/models/Qwen3.8-27B-FP8`

## Decision

Do not enable TurboMind MTP by default. Keep `num_draft_tokens=0`.

The implementation now survives mixed-row retirement and restores recurrent
state correctly, but exact greedy MTP is slower than ordinary decoding at both
depths tested:

| Depth | Committed tokens / verification forward | Decode throughput vs K=0 | Decision |
| ---: | ---: | ---: | --- |
| K=1 | 0.62 / 2 | 0.552x | reject |
| K=4 | 0.02 / 5 | 0.288x | reject |

K=1 improved only from 0.538x to 0.552x after removing the temporary
per-draft CUDA synchronizations. The remaining loss is structural rather than
a hidden synchronization tax.

## Correctness findings

### Recurrent rollback

A rejected verification must restore 48 Gated DeltaNet layers. The retained
implementation:

- snapshots convolution and recurrent state before verification;
- restores only rows that reject;
- publishes no token on a rejecting recurrent row;
- forces one ordinary decode before drafting again;
- keeps snapshot frontier metadata bounded to the active batch;
- handles accepted EOS without appending the verifier bonus.

A diagnostic fingerprint of every recurrent and convolution-state byte matched
before verification and before ordinary replay on all four TP ranks. This
establishes byte-exact GDN restoration for the traced transitions.

Artifact:
`/localpool/lmdeploy-v100-next/results/20260827_222055-spectrace-20616e837294`

### Fresh-process token equality is not a valid oracle

The K=0 control is not byte-deterministic across fresh processes. K=0 versus
K=0 repeatedly flipped row 2 at output position 3 (`4087` versus `5707`), under:

- ordinary execution;
- `CUDA_LAUNCH_BLOCKING=1`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Therefore a fresh-process K=0 versus K=N text mismatch at that tie cannot be
attributed to speculation.

### Same-process verifier/replay numeric contract

`TM_SPEC_LOGIT_PARITY=1` retained the verifier logits before forced rejection
and compared them with the next ordinary replay after restoration:

- comparisons: 15;
- maximum absolute logit error: 0.05078125;
- maximum RMS logit error: 0.0075805338;
- same argmax: 14/15;
- only argmax split: `5707` versus `4087`;
- both cross-scores at the split: 26.953125 in FP16;
- split comparison max absolute error: 0.015625;
- split comparison RMS error: 0.0028913227.

The observed token split is an FP16 near-tie, not state-publication drift.

Artifact:
`/localpool/lmdeploy-v100-next/results/20260827_225119-spectrace-29427f235dc3`

## Performance evidence

### K=1, final hot path

Commit: `f1268a4dd5f9`

- K=0 mean decode: 54.03 tok/s;
- K=1 mean decode: 29.82 tok/s;
- ratio: 0.552x;
- K=1 committed 476 tokens over 768 verification forwards in the measured
  acceptance stream (238 full accepts).

Artifact:
`/localpool/lmdeploy-v100-next/results/20260827_225801-specverify-f1268a4dd5f9`

### K=4

Commit: `f3d9b3719a65`

- ratio: 0.288x;
- committed rate: 0.02 tokens per verification forward;
- 20 tokens committed over 960 verification forwards (4 full accepts).

Artifact:
`/localpool/lmdeploy-v100-next/results/20260827_215413-specverify-f3d9b3719a65`

## Why exact MTP cannot pay on this path

For this recurrent model, a partial draft prefix cannot be retained after a
rejection without an intermediate recurrent-state snapshot or replay. The
current safe policy is all-or-nothing:

- full acceptance commits K drafts plus the verifier bonus;
- any rejection commits nothing and requires an ordinary decode next.

At K=1 the benchmark's full-accept probability is about 31%. Even before draft
cost, that policy advances only `(1+p)/(2-p) ~= 0.78` token per target forward
relative to ordinary decoding. The MTP layer and shared LM-head projection add
more work. K=4 compounds draft error and almost never accepts the entire run.

Removing two temporary stream synchronizations per K=1 attempt improved the
ratio by only 0.014, confirming that synchronization was not the dominant
limit.

## Conditions for reopening

Do not resume by micro-optimizing the current all-or-nothing loop. Reopen only
with one of these designs:

1. capture the GDN state after the verifier processes the committed tip, so a
   rejection can commit the verifier token without an ordinary replay;
2. provide per-position recurrent snapshots cheaply enough to retain accepted
   prefixes;
3. add a selective-drafting policy proven to exceed the break-even full-accept
   probability on a representative serving workload;
4. use an explicitly approved approximate acceptance policy with a separate
   quality qualification.

Any reopened path must rerun same-process logit parity, mixed-row retirement,
EOS, forced-rejection, and end-to-end throughput gates. A throughput ratio at
or below 1.0 is a rejection regardless of acceptance telemetry.
