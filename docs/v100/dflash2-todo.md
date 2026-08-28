# DFlash2 V100 fix backlog

Status: active  
Last updated: 2026-08-28  
Companion evidence log: [`dflash2-investigation.md`](./dflash2-investigation.md)

This is the accumulating, prioritized list of DFlash2 findings to fix or falsify. A checked item means the implementation and its required gates passed; it does not imply the overall DFlash2 path is qualified.

## Exit criteria

DFlash2 is qualified only when all of the following hold on the exact audited 1,000-token prompt:

- exact K=0/K=7 output identity passes;
- average committed length approaches the SGLang reference of 3.765;
- decode throughput provides a material uplift over LMDeploy K=0, targeting approximately 2.2x;
- TP4 communication remains NCCL over the fully connected NVLink island;
- mixed-row retirement, EOS, partial-prefix, rejection, and ambiguity paths pass.

## P0: acceptance and draft fidelity

- [x] **Remove the unmatched BF16 round after context `hidden_norm`.**
  - The no-round path matches SGLang, passed exact audited-prompt identity, and is now the default.
  - Its measured acceptance effect was small: roughly +0.04 committed tokens/step.

- [x] **Expose raw acceptance separately from post-replay commits.**
  - Commit `720c8e70` reports raw accepted drafts, raw commit length, ambiguous verifications, and accepted tokens discarded by ambiguity replay.

- [x] **Run the four-arm context-round/ambiguity matrix.**
  - Context round had little effect: raw commit remained approximately 2.1.
  - Removing the 0.0625 ambiguity margin raised final commit from 1.83-1.87 to 2.05-2.09 and throughput from 42-43 to about 47.4 tok/s.
  - Every arm passed the 256-token short-prompt identity gate; the selected no-round/zero-margin arm also passed exact audited-prompt identity.

- [x] **Inspect the draft checkpoint architecture and unmatched weight keys.**
  - The checkpoint is generic `DFlash2DraftModel`, not Laguna.
  - Its 81 tensors contain no `aux_hidden_norms` or `g_proj`; the generic loader is the correct architecture.

- [x] **Load attention type and window per draft layer.**
  - The loader now configures homogeneous sliding/full policies correctly and rejects unsupported mixtures.
  - The corrected policy passed exact audited identity but did not improve acceptance: raw commit 2.033 corrected versus 2.077 legacy.

- [ ] **Move TP all-reduces before output grouped convolution and W2 row-scale restoration.**
  - Classification: confirmed TP4 operation-order mismatch.
  - Attention: SGLang performs local Wo, FP16 all-reduce, then output grouped convolution; LMDeploy performs local Wo, output convolution, then all-reduce inside residual norm.
  - MLP: SGLang performs local W2, FP16 all-reduce, row-scale restoration, then output grouped convolution; LMDeploy restores row scale and applies output convolution before all-reduce.
  - These operations commute only in exact arithmetic; FP16 stores and collectives create ten mismatched rounding boundaries across five layers.
  - The first A/B showed no fidelity benefit: raw commit 2.107 old versus 2.110 reduce-first; reduce-first throughput was 46.16 versus 47.18 tok/s. Short identity passed both.
  - Keep open only as a tensor-parity discrepancy; it is not the current acceptance lead.
  - Done when: shared-tensor parity proves the required boundary and any adopted order avoids the measured regression.

- [ ] **Write only committed verifier context K/V, or prove rejected suffixes are invisible.**
  - Classification: lifecycle hypothesis.
  - SGLang uses `commit_lens` and prefix-valid cache writes. LMDeploy writes all verifier positions before acceptance and relies on logical lengths plus later overwrite.
  - Experiment: poison rejected context-KV suffixes and verify that the next draft is unchanged.
  - Done when: committed-prefix writes are implemented or the poison test proves the suffix cannot affect drafting/cache reuse.

- [ ] **Validate draft attention metadata after partial acceptance.**
  - Classification: strongest unconfirmed lifecycle hypothesis.
  - Assert for every row that the draft key span equals the newly published committed length plus block size.
  - Control: rebuild DFlash Setup/Prepare metadata after Rollback and compare candidates and acceptance.
  - Done when: metadata is refreshed correctly or the control falsifies this cause.

- [ ] **Add first-block tensor parity against SGLang.**
  - Compare, in order: captured target features, context FC output, context norm output, each grouped-convolution output, attention output, residual norm, final draft hidden state, candidate IDs, unary scores, and selector scores.
  - Done when: the first numerical divergence is identified and either fixed or documented as intentional.

- [ ] **Validate selector edge-score narrowing semantics.**
  - Classification: runtime-codegen-dependent hypothesis.
  - SGLang expresses `predecessor * hidden` as an FP16 tensor before contraction; LMDeploy promotes all three factors and accumulates serially in FP32.
  - Compare every interaction score and top-two edge margin from shared tensors; inspect SGLang's generated kernel before changing LMDeploy.
  - Done when: generated arithmetic is proven equivalent or LMDeploy matches the observed narrowing points.

- [x] **Fix draft RoPE ownership and validate post-RoPE K parity.**
  - All DFlash layers had inherited target partial multimodal RoPE instead of their own full 128-dimensional standard RoPE.
  - Per-layer RoPE raised commit length from 2.033 to 2.664 and decode from 46.98 to 61.39 tok/s.
  - The generation-limit publication defect exposed by higher acceptance was fixed; exact audited K=0/K=7 identity passed for 256 tokens.

- [ ] **Run isolated grouped-convolution arithmetic parity.**
  - Classification: lower-priority rounding hypothesis; indexing already appears aligned.
  - Compare both convolution sides bitwise from identical FP16 inputs, deltas, and base kernels.
  - Done when: arithmetic matches or required FP16/FP32 narrowing points are identified and reproduced.

- [ ] **Compare residual RMSNorm reduction and rounding schedules.**
  - Classification: deferred lower-level hypothesis.
  - Investigate only if first-block parity first diverges at a residual norm after earlier boundaries match.
  - Done when: the V100 SGLang kernel's reduction/rounding schedule is matched or shown immaterial.

## P1: speculative cycle cost

- [ ] **Capture fixed-shape target verification and draft execution in CUDA graphs.**
  - Nsight shows substantial target submission/synchronization time around the eight-token verification shape.
  - First make allocations and addresses stable; then capture batch-size-one K=7 paths.
  - Done when: graph replay passes identity and matched profiling shows reduced cycle wall time.

- [ ] **Eliminate mandatory draft candidate device-to-host synchronization.**
  - Keep candidate IDs and accepted-prefix decisions on device through verification setup where possible.
  - Done when: no per-cycle host synchronization is required merely to publish seven draft IDs.

- [ ] **Make target-verification temporary buffers stable.**
  - Nsight reports hundreds of async allocations/frees and kernel submissions per cycle.
  - Preallocate or arena-allocate fixed K=7/batch-one workspaces as a CUDA-graph prerequisite.
  - Done when: allocator calls disappear from steady-state verification traces.

- [ ] **Combine rejection argmax and ambiguity detection into one vocabulary pass.**
  - The rejection kernel currently scans every target-logit row once for argmax and again for ambiguity.
  - Preserve deterministic score-descending/token-ID-ascending behavior.
  - Done when: exact identity passes and the rejection kernel time falls from the profiled baseline.

- [ ] **Evaluate the draft attention kernel against SGLang's V100 backend.**
  - Draft attention remains a material GPU component.
  - Compare equivalent shapes and isolate kernel efficiency from launch overhead.
  - Done when: either LMDeploy matches the reference kernel cost or a concrete replacement is implemented.

## Completed fixes

- [x] **Use the exact audited SGLang prompt IDs.**
  - Prompt length 1,000; SHA-256 `9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01`.

- [x] **Support partial-prefix recurrent-state commit.**
  - Intermediate GDN states are selected at the accepted frontier rather than requiring all-or-nothing acceptance.

- [x] **Match Laguna wide-branch and W13 scaling.**
  - Includes pre-Wo 1/256 scaling, W13 input scaling, BF16-compatible SwiGLU, and dynamic W2 row scaling. The row-scale collective order remains an open item above.

- [x] **Use FP16 NCCL for scaled draft branches.**
  - Removed unnecessary FP16-to-FP32 cast kernels and extra collective traffic.

- [x] **Require TP4 NCCL over the NVLink island.**
  - Topology gate and NCCL logs confirm `P2P/direct pointer`, `PXN 0`, and no host/network fallback.

- [x] **Replace full-vocabulary TP exchange with local top-16 exchange.**
  - Exact identity passed.

- [x] **Replace sixteen vocabulary scans with one-pass top-16.**
  - Exact identity passed; combined selector work improved audited throughput materially.

- [x] **Parallelize selector candidate scoring.**
  - Preserves each candidate's original accumulation order and passes identity.

- [x] **Skip discarded attention and Wo during context-KV materialization.**
  - Matches SGLang's K/V-only intent and passes identity.

- [x] **Add matched Nsight/NVTX profiling.**
  - Current ranges include target verification, rollback, draft/select, rejection, target decode, and context-KV materialization.

## Current measured state

- LMDeploy DFlash2 audited decode: 61.39 tok/s
- LMDeploy committed length: 2.664
- SGLang committed length: 3.765
- Exact short-workload identity: pass
- Cumulative audited gain from the original 36.13 tok/s: about 70%

The immediate execution order is: validate selector score narrowing, capture first-block tensor parity, then investigate context-KV/metadata lifecycle if the fidelity gap remains.
