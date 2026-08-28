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

- [ ] **Remove the unmatched BF16 round after context `hidden_norm`.**
  - Classification: confirmed semantic mismatch.
  - LMDeploy explicitly BF16-rounds `DFlashPredictor::ProjectContext`; SGLang uses `LagunaRMSNorm(..., scaled_residual_stream=False)` and ordinary FP16 RMSNorm at this boundary.
  - Experiment: audited A/B with context round on/off and no other change.
  - Done when: the matching no-round path passes identity and its acceptance/throughput effect is recorded.

- [ ] **Expose raw acceptance separately from post-replay commits.**
  - Classification: missing diagnostic for a confirmed policy difference.
  - Add counters for raw accepted drafts, raw commit length, ambiguous verifications, zero-commit replays, and accepted drafts discarded by ambiguity replay.
  - Done when: every audited result reports raw and final acceptance.

- [ ] **Run the four-arm context-round/ambiguity matrix.**
  - Arms: context round on/off crossed with ambiguity margin `0.0625/0`.
  - Preserve an exact K=0 identity arm for each configuration.
  - Done when: one table attributes fidelity loss to draft math versus exactness replay.

- [ ] **Inspect the draft checkpoint architecture and unmatched weight keys.**
  - Classification: high-impact conditional mismatch.
  - SGLang has a `DFlashLagunaForCausalLM` contract with `aux_hidden_norms`, per-layer input norms for context K/V, and `g_proj`; LMDeploy always constructs its generic DFlash2 model.
  - Record `architectures`, `layer_types`, and all unloaded/unmatched checkpoint keys.
  - Done when: the checkpoint is proven generic or every Laguna-specific weight and operation is supported.

- [ ] **Load attention type and window per draft layer.**
  - Classification: confirmed loader mismatch; workload effect pending config inspection.
  - SGLang resolves full versus sliding attention per layer. LMDeploy currently assigns one global causal/window configuration and ignores `layer_types`.
  - Done when: every loaded TurboMind draft layer matches SGLang's per-layer attention type and `sliding_window - 1` convention.

- [ ] **Restore dynamic W2 row scales after TP all-reduce.**
  - Classification: confirmed operation-order mismatch when a row scale exceeds one.
  - SGLang reduces W2 output first and restores the row scale afterward. LMDeploy currently scales each rank's partial W2 output before the collective.
  - First diagnostic: report the row-scale distribution on the audited prompt.
  - Done when: order matches SGLang, identity passes, and acceptance impact is recorded.

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

- LMDeploy DFlash2 audited decode: 42.27 tok/s
- LMDeploy committed length: 1.862
- SGLang committed length: 3.765
- Exact short-workload identity: pass
- Cumulative audited runtime gain from completed selector/context work: about 17%

The immediate execution order is: context-round A/B, raw acceptance counters, four-arm ambiguity matrix, checkpoint architecture/keys, then W2 collective ordering and per-layer attention configuration.
