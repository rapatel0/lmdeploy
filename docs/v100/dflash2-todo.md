# DFlash2 V100 fix backlog

Status: active
Last updated: 2026-08-29
Companion evidence log: [`dflash2-investigation.md`](./dflash2-investigation.md)
Execution plan: [`dflash2-performance-plan.md`](./dflash2-performance-plan.md)

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
  - Disabling replay for remaining exact ties passed two of three audited controls; the third was the known position-220 split. Separate-process throughput was lower rather than higher, so exact-tie replay remains enabled by default.

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

- [x] **Add first-block tensor parity against SGLang.**
  - Commit `b753831d` produced four complete 102-boundary SGLang manifests for the exact forced block `[1144, 248070 x 7]` at positions `1000..1007`; TurboMind block IDs and embeddings match exactly.
  - Capture boundaries and feature order are semantically aligned. Target-residual RMS drift grows across layers `[5, 19, 33, 47, 61]` as `0.1075`, `0.3547`, `0.6800`, `1.3486`, and `4.6666`.
  - Commit `4fe99716` replayed SGLang's exact target residual inside TurboMind across TP4. Context FC RMS fell from `44.2053` to `0.05926`, and normalized context passed parity at max abs `0.00390625`, RMS `0.000277`.
  - The context projector is functionally aligned after normalization; the dominant context mismatch is upstream target-model numerical trajectory drift. Artifacts: `/results/20260829_032508-sglang-dflash-parity-b753831db680` and `/results/20260829_035037-dflash-context-replay-4fe9971622bc`.

- [x] **Validate selector edge-score narrowing semantics.**
  - Explicitly FP16-rounding `predecessor * hidden` produced the same 2.664 commit length and identical acceptance counts as FP32 intermediates.
  - Throughput was 61.14 versus 61.76 tok/s, within run variance. The experimental branch was removed; this is not the fidelity gap.

- [x] **Validate grouped-convolution FP16 operation boundaries.**
  - Explicit FP16 coefficient, product, and accumulation boundaries produced the same 2.664 commit length and identical counters as FP32 accumulation.
  - Throughput was 61.40 versus 60.90 tok/s; the FP16 arm passed audited identity. The branch was removed.

- [x] **Fix draft RoPE ownership and validate post-RoPE K parity.**
  - All DFlash layers had inherited target partial multimodal RoPE instead of their own full 128-dimensional standard RoPE.
  - Per-layer RoPE raised commit length from 2.033 to 2.664 and decode from 46.98 to 61.39 tok/s.
  - The generation-limit publication defect exposed by higher acceptance was fixed; exact audited K=0/K=7 identity passed for 256 tokens.

- [ ] **Run isolated grouped-convolution arithmetic parity.**
  - Classification: lower-priority rounding hypothesis; indexing already appears aligned.
  - Compare both convolution sides bitwise from identical FP16 inputs, deltas, and base kernels.
  - Done when: arithmetic matches or required FP16/FP32 narrowing points are identified and reproduced.

- [ ] **Compare residual RMSNorm reduction and rounding schedules.**
  - Classification: active draft-fidelity mismatch.
  - Same-input parity makes block embeddings bit-identical, then first diverges at `block.initial_norm`: max abs `0.015625`, RMS `0.000728`.
  - SGLang's SM70 Laguna RMSNorm multiplies normalized activations by BF16-rounded weights in FP32 before BF16 output rounding. TurboMind's generic non-zero-centered RMSNorm narrows the normalized activation to FP16 before multiplying the weight, then rounds the result to BF16.
  - Next experiment: a flagged TurboMind full-product/BF16-output initial RMSNorm arm, using the same build for parity, five-trial acceptance, and audited identity.
  - Done when: the V100 SGLang kernel's reduction/rounding schedule is matched or shown immaterial.

## P1: speculative cycle cost

- [x] **Attribute one complete speculative cycle with Nsight Systems.**
  - Commit `930baf48` measured 48.6 ms per K=7 cycle on the audited prompt.
  - NVTX attributed 15.02 ms to target verification, 23.54 ms to rollback, 7.40 ms to draft/selection, and 0.09 ms to rejection.
  - These ranges plus context-KV work explain about 96% of normalized cycle time.
  - K=7 spent 59.1% of aggregate CUDA API time inside 25,332 `cudaMemcpyAsync` calls.
  - The hot speculative readbacks use pageable host buffers, so pinned staging is the first measured host-control experiment.

- [x] **Test persistent pinned staging for speculative readbacks.**
  - Five-trial acceptance-normalized cycle time was 43.04 ms pageable, 42.92 ms pinned, and 42.99 ms pinned plus one combined rollback barrier.
  - The largest change was 0.3%, so the apparent `cudaMemcpyAsync` API cost was outstanding GPU work at the synchronization boundary rather than removable pageable staging overhead.
  - Exact audited identity passed. The experimental hot-path buffers and controls were removed.

- [x] **Tune SM70 small-Q attention tiles for the verifier and draft.**
  - Fresh Nsight attribution: target head-dim-256 verification attention is 11.9% of K=7 GPU time; all draft head-dim-128 attention is 1.2%.
  - The first matrix suggested about 1% normalized cycle gains, but the five-trial confirmation falsified them: Q64/KV64 took 43.94 ms/step versus 44.17 ms/step for combined draft-Q16/target-Q32.
  - The alternate kernels were removed; Q64/KV64 remains the default. Tile size alone does not reproduce SGLang's grouped/split attention design.

- [ ] **Capture fixed-shape target verification and draft execution in CUDA graphs.**
  - Selector-only capture now succeeds on all four TP ranks with thread-local capture.
  - The selector-only slice is rejected: profiled cycle time regressed from 47.20 to 48.17 ms, while `dflashDraftAndSelect` remained 7.071 versus 7.068 ms.
  - The full draft-plus-selector graph captured/replayed on all four TP ranks and passed audited identity.
  - It reduced `dflashDraftAndSelect` from 6.99 to 6.60 ms and kernel launches from 193,044 to 166,932, but whole-cycle gain was only 0.6% unprofiled and regressed under Nsight.
  - Keep it off by default as infrastructure for broader capture.
  - A per-layer target-attention graph captured/replayed all 64 rank/layer instances and reduced `targetVerify` from 14.235 to 13.904 ms, but sixteen graph launches per verification caused whole profiled request time to regress by 0.8%.
  - The per-layer target graph was removed. Future target capture must span a larger contiguous region and amortize graph-launch overhead.
  - Done when: broader graph replay passes identity and reduces matched whole-cycle wall time by more than run variance.

- [ ] **Qualify direct paged Q=8 attention.**
  - The SM70 block-iterator path now bypasses flattened KV and runs on TP4.
  - Allocator/free calls fell from 90,476 to 86,192.
  - Unprofiled normalization improved from 42.66 to 41.61 ms, but profiled normalization regressed from 47.20 to 47.64 ms.
  - Repeated exactness produced one paged pass plus known position-145/220 splits; flattened controls hit the same positions and had no pass in three attempts.
  - Flattened-versus-paged and flattened-versus-flattened parity both first diverged at the pre-attention target residual and changed intermediate candidate IDs, while final selected IDs remained exact.
  - Keep the path off by default and use it only as the current full-graph prerequisite.
  - Done when: full-graph qualification proves a net benefit, or the graph is redesigned around flattened KV and this path is removed.

- [ ] **Eliminate mandatory draft candidate device-to-host synchronization.**
  - Phase-owned host frontiers now remove the redundant pre-draft sequence-length copy/synchronization and retain a safe legacy fallback.
  - The matched profile removed 236 copies but regressed whole request time by 0.5%; retain this only as infrastructure, not a standalone speed claim.
  - Keep candidate IDs and accepted-prefix decisions on device through verification setup where possible.
  - Done when: no per-cycle host synchronization is required merely to publish seven draft IDs.

- [ ] **Make target-verification temporary buffers stable.**
  - Phase-owned TurboMind workspaces now cover context projection, embeddings, draft residual/convolution/MLP tensors, proposal/selector tensors, and—at commit `d277060f`—local LM-head and TP top-16 exchange buffers.
  - The first profiled slice reduced steady allocator/free calls from 108,040 to 94,556 per captured aggregate. The expanded top-16 and UnifiedAttention qkv/output/flattened-KV workspaces reduced them further to 90,476, a total 16.3% reduction.
  - Four-rank parity and exact audited identity passed. Matched profiling improved normalized cycle time by 0.7%; five-trial unprofiled normalization improved by 1.6%.
  - Target-verification and lower-level library workspaces remain.
  - Done when: allocator calls disappear from steady-state verification traces.

- [x] **Combine rejection argmax and ambiguity detection into one vocabulary pass.**
  - A deterministic top-2 block reduction preserves score-descending/token-ID-ascending order and removes the second full-vocabulary scan.
  - The profiled rejection kernel fell from 1.495 to 0.971 ms average, a 35% kernel reduction.
  - Matched K=7 profile cycle time improved from 48.08 ms with workspaces to 47.17 ms with one-pass rejection; exact audited identity and four-rank parity capture passed.
  - One-pass rejection is now the default; `TM_DFLASH_ONE_PASS_REJECT=0` retains the legacy control.

- [ ] **Evaluate the draft attention kernel against SGLang's V100 backend.**
  - Draft attention remains a material GPU component.
  - Compare equivalent shapes and isolate kernel efficiency from launch overhead.
  - Done when: either LMDeploy matches the reference kernel cost or a concrete replacement is implemented.

## Completed fixes

- [x] **Parallelize the local top-16 merge with CUB.**
  - Replaced the one-thread merge of 256 lane-local top-16 lists with `cub::BlockReduce`.
  - Five-trial acceptance-normalized cycle time improved from 43.87 to 43.45 ms (about 1%); audited identity passed.
  - CUB is the default, with `TM_DFLASH_CUB_TOPK=0` as the legacy control.

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

Run two tracks in parallel. Remove host barriers and stabilize workspaces on Track A. Capture first-block tensor parity on Track B.
