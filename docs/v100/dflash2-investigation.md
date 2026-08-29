# DFlash2 V100 investigation

Status: active investigation
Last updated: 2026-08-29
Target: Qwen3.8-27B-FP8, TP4 V100, DFlash2 block size 8

Action tracker: [`dflash2-todo.md`](./dflash2-todo.md)
Execution plan: [`dflash2-performance-plan.md`](./dflash2-performance-plan.md)

## Qualification target

Match the audited SGLang workload and approach its measured speculative result:

- Exact prompt: 1,000 token IDs, SHA-256 `9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01`
- SGLang target-only reference: 58.21 decode tok/s
- SGLang DFlash2 reference: 136.6 decode tok/s
- SGLang average commit length: 3.765 tokens per verification
- Desired uplift: approximately 2.2x or better

LMDeploy must also preserve exact target-only output identity.

## Current result

Latest qualified audited LMDeploy result:

- DFlash2 decode: 61.39 tok/s
- Average committed length: 2.664
- Verification cycle time: approximately 43.4 ms
- Exact audited K=0/K=7 identity: pass
- Original audited DFlash2 result: 36.13 tok/s
- Cumulative runtime improvement: about 70%

This remains unqualified. SGLang commits 3.765 tokens per step and completes each cycle in approximately 27.6 ms.

## Confirmed runtime findings

### NVLink and NCCL transport

The benchmark uses physical GPUs 4-7. Every pair is connected by NV1 or NV2. The launcher now aborts on any non-NVLink edge and explicitly selects the NCCL communicator.

Verified NCCL initialization reported:

- `P2P/direct pointer` for every channel
- `PXN 0`
- no host, socket, IB, or cross-NUMA model-execution transport

Relevant commit: `2fa1e9e5` (`Require NVLink NCCL benchmark transport`).

### Full-vocabulary candidate exchange

LMDeploy originally projected and gathered all `7 x 248K` vocabulary logits across TP4 before selecting 16 candidates. SGLang computes top-16 per rank, gathers only those candidates, and merges them globally.

Implemented:

- TP-local LM-head projection
- local top-16
- NCCL exchange of only candidate IDs and scores
- deterministic global top-16 merge

The first implementation raised audited throughput from 36.13 to 40.89 tok/s. It passed exact identity.

Relevant commits:

- `5d1af4e4` (`Gather only DFlash top-k candidates`)
- `3a19f903` (`Support int32 NCCL collectives`)
- `5a5dbd62` (`Enable sharded DFlash selection by default`)

### Sixteen redundant vocabulary scans

The top-16 CUDA kernel reread every vocabulary shard 16 times, once per selected rank. Nsight measured it at approximately 1.58 ms per speculative cycle.

It now reads each score once, retains per-lane top-16 lists, and performs a deterministic 256-way merge. Exact identity passed. After this change:

- full-vocabulary path: 39.71 tok/s
- sharded candidate path: 40.57 tok/s

Most of the selector gain came from removing the repeated scans; the reduced NCCL payload contributes roughly another 2%.

Relevant commit: `6dd39cc8` (`Select DFlash top-k in one vocabulary pass`).

### Discarded context attention

LMDeploy refreshed draft context KV by running full attention and Wo for all five DFlash layers, then discarding every output. SGLang uses `kv_proj_only`: it projects and normalizes K/V, applies RoPE, and writes the cache without attention or Wo.

LMDeploy now retains the identical QKV/KV-write path but skips flattening, attention, output gating, and Wo where the output is unused. Selector candidates are also evaluated concurrently while preserving each candidate's original FP32 accumulation order.

Results:

- previous audited result: 40.57 tok/s
- KV-only + parallel selector: 42.27 tok/s
- incremental gain: about 4.2%
- exact identity: pass

Relevant commit: `e1c27ba2` (`Skip discarded DFlash context attention`).

## Nsight phase measurements

Before KV-only context materialization, NVTX measured average host ranges per rank:

| Range | Average |
| --- | ---: |
| `targetVerify` | 18.60 ms |
| `speculativeRollback` | 20.43 ms |
| `dflashDraftAndSelect` | 8.17 ms |
| `targetDecode` | 19.37 ms |
| `speculativeReject` | 0.11 ms |

After KV-only context materialization:

| Range | Average |
| --- | ---: |
| `targetVerify` | 15.43 ms |
| `speculativeRollback` | 22.77 ms |
| `dflashDraftAndSelect` | 8.20 ms |
| `targetDecode` | 15.95 ms |
| `dflashContextKV` | 0.50 ms |
| `speculativeReject` | 0.10 ms |

`speculativeRollback` begins with device-to-host verdict copies followed by a stream synchronization. Its time therefore includes outstanding target-verification and context-update tail work; it must not be added to kernel totals as independent computation.

The profile shows both substantial target GPU work and a large submission/synchronization component. CUDA graphs remain a plausible route to SGLang's lower cycle cost, but they cannot compensate for the current acceptance deficit.

Profile artifacts:

- `/localpool/lmdeploy-v100-next/results/20260828_181330-nsys-dflash-5a5dbd6292fc`
- `/localpool/lmdeploy-v100-next/results/20260828_182620-nsys-dflash-6dd39cc8f339`
- `/localpool/lmdeploy-v100-next/results/20260828_183837-nsys-dflash-e1c27ba2940a`

## Confirmed fidelity mismatch awaiting A/B

### Extra BF16 rounding after context hidden norm

LMDeploy `DFlashPredictor::ProjectContext` currently performs:

1. context FC projection;
2. RMSNorm;
3. explicit BF16 round retained in FP16 storage.

SGLang constructs `hidden_norm` as `LagunaRMSNorm(..., scaled_residual_stream=False)`. With a residual scale of one, `LagunaRMSNorm` delegates to ordinary FP16 RMSNorm and performs no Laguna BF16 residual rounding.

Therefore LMDeploy changes every projected context vector before building all five layers' draft K/V. This is a confirmed semantic mismatch and is capable of reducing draft fidelity across the entire seven-token block.

The 2026-08-28 four-arm audited matrix showed that removing this round is semantically correct but not the dominant acceptance fix:

- with the 0.0625 ambiguity margin, commit length changed from 1.832 to 1.871 and decode from 42.26 to 43.17 tok/s;
- with zero ambiguity margin, commit length changed from 2.050 to 2.087 and decode from 47.46 to 47.39 tok/s;
- raw commit length remained approximately 2.1 in every arm.

All four arms passed the existing 256-token short-prompt K=0/K=7 identity gate. The selected no-round/zero-margin configuration subsequently passed exact K=0/K=7 identity for 256 generated tokens on the audited 1,000-token prompt (`DFLASH_AUDITED_IDENTITY_PASS`, artifacts `/results/20260828_191101-dflash-audited-identity-f5868503af67`).

## Confirmed TP4 draft-network mismatch

### Collective boundary is on the wrong side of output convolution

SGLang performs, for attention:

`local Wo -> FP16 TP all-reduce -> output grouped convolution -> residual norm`

LMDeploy performs:

`local Wo -> output grouped convolution -> FP16 TP all-reduce inside residual norm`

For MLP, SGLang likewise reduces raw W2 first, then restores dynamic row scale and applies the output convolution. LMDeploy restores the scale and convolves each rank's partial output before reduction.

These forms commute in exact arithmetic but not across FP16 stores and FP16 NCCL. However, the audited A/B falsified this as an explanation for low acceptance: raw commit was 2.107 with the old order and 2.110 with reduce-first, while decode fell from 47.18 to 46.16 tok/s. The short identity gate passed both arms. Keep this as a tensor-parity discrepancy, not an acceptance lead.

## Confirmed acceptance-accounting amplifier

### Whole-block ambiguity replay

The rejection kernel marks a path ambiguous when a competing target logit lies within the configured ambiguity margin (currently 0.0625). For DFlash, `LanguageModel::Rollback` treats any ambiguous path as a no-commit event:

- restore the recurrent state;
- discard every otherwise accepted draft and verifier bonus;
- run an ordinary target decode next.

Commit `720c8e70` added raw-versus-final counters. The audited matrix measured:

| Context BF16 round | Ambiguity margin | Final commit | Raw commit | Ambiguous steps | Discarded tokens | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0 | 2.087 | 2.087 | 0 | 0 | 47.39 |
| off | 0.0625 | 1.871 | 2.116 | 36 | 93 | 43.17 |
| on | 0 | 2.050 | 2.101 | 9 | 18 | 47.46 |
| on | 0.0625 | 1.832 | 2.092 | 45 | 99 | 42.26 |

Thus the 0.0625 policy discards roughly 0.22-0.26 tokens per verification and costs about 4-5 tok/s. It explains a meaningful part of the low published acceptance, but raw fidelity is still only about 2.1 versus SGLang's 3.765. The remaining gap is predominantly draft-network fidelity, not accounting.

A zero margin can still flag exact ties, as seen in the round-on arm. The no-round/zero-margin configuration passed exact audited-prompt identity, so the default margin was changed to zero while preserving the environment override for conservative diagnostics.

A later exact-tie replay A/B did not qualify disabling replay as a default. Replay-on and replay-off ran in separate processes and followed different acceptance trajectories: replay-on had no ambiguous steps, commit length 2.669, and 63.28 decode tok/s; replay-off had 12 ambiguous steps, commit length 2.547, and 60.23 tok/s. Replay-off passed audited identity twice and hit only the known position-220 split once, so the control remains useful, but there is no matched evidence of a speed uplift. Exact-tie replay remains enabled by default. Artifacts: `/results/20260829_001752-dflash-exact-tie-79fc95e78c15`.

## Confirmed checkpoint attention-policy defect

The published checkpoint declares:

- `architectures=["DFlash2DraftModel"]`;
- five `layer_types`, all `sliding_attention`;
- `sliding_window=2048`;
- top-level `is_causal=false`.

SGLang deliberately interprets every DFlash `sliding_attention` layer as decoder/causal attention regardless of the top-level flag, with 2047 visible tokens to the left. LMDeploy ignored `layer_types` and copied `is_causal=false`, so all five trained causal draft layers ran as non-causal attention. Depending on the TurboMind kernel specialization, this can also defeat enforcement of the intended left window.

The checkpoint is the generic DFlash2 architecture, not Laguna: its 81 tensor keys contain no `aux_hidden_norms` or `g_proj`. Thus Laguna-specific missing weights are not an active issue for this model.

The loader now validates `layer_types`, configures homogeneous sliding layers as causal/windowed and homogeneous full layers as encoder-only, and rejects unsupported mixed policies instead of silently flattening them. The audited A/B showed that this correctness fix is not an acceptance driver: raw commit was 2.077 with the legacy policy and 2.033 with the corrected policy, while decode was 47.22 versus 47.34 tok/s. The corrected policy passed exact audited identity.

## Confirmed draft RoPE defect

`UnifiedAttentionLayer` initialized one global RoPE configuration from `weights[0]` and used it for every target and DFlash attention layer. The target and draft contracts differ materially:

- target: head dimension 256, partial rotary factor 0.25, multimodal/interleaved RoPE;
- draft: head dimension 128, full standard RoPE;
- both use theta 10,000,000, but dimension and layout are not interchangeable.

Consequently all five DFlash layers used target RoPE rather than the draft checkpoint's own `AttentionWeight::rope`. The implementation now builds one `RopeKernelParam` per attention weight and selects it by the actual weight object.

The audited A/B confirmed a large uplift:

| RoPE policy | Commit length | Decode tok/s |
| --- | ---: | ---: |
| target-global legacy | 2.033 | 46.98 |
| per-layer checkpoint RoPE | 2.664 | 61.39 |

This is a 31% acceptance gain and 30.7% throughput gain over the matched legacy arm. Higher acceptance exposed a latent terminal-state defect: a verification reaching `max_seq_len` did not run Generation's ordinary stop criterion, so it could schedule one more step. The engine now clamps publication defensively and verification explicitly marks limit-reaching rows finished. The rebuilt audited gate passed exact K=0/K=7 identity for 256 generated tokens (`DFLASH_AUDITED_IDENTITY_PASS`, artifacts `/results/20260828_195358-dflash-audited-identity-7eaf894d0dee`).

## Selector rounding A/B

Explicitly FP16-rounding the selector's `predecessor * hidden` intermediate did not change the audited result: both arms committed 2.664 tokens/step with identical accepted-draft and full-accept counts. Decode measured 61.14 tok/s with the original FP32 intermediate and 61.76 tok/s with explicit narrowing, which is within run variance. This hypothesis is closed and the extra branch was removed.

## Grouped-convolution rounding A/B

Matching PyTorch's apparent FP16 operation boundaries inside grouped convolution did not change acceptance: both the original FP32-accumulation arm and the FP16-step arm committed 2.664 tokens/step with identical counters. Decode was 61.40 versus 60.90 tok/s, respectively. The FP16-step arm passed the audited 256-token K=0/K=7 identity gate. This hypothesis is closed and the experimental branch was removed.

## Corrected-path Nsight profile

A fresh Nsight Systems capture at commit `948484b9` localized the current K=7 GPU time. The generic SM70 target-verification attention kernel (`head_dim=256`, Q64/KV64) consumed **11.9%** of aggregate GPU kernel time, while all five draft `head_dim=128` attention calls together consumed only **1.2%**. The target verifier therefore has much more tile-tuning headroom than the draft attention itself. Full/fragmented GEMMs remained dominant, and the trace also confirmed substantial verifier rejection, selector, NCCL, and recurrent-kernel costs. Artifacts: `/results/20260828_201555-nsys-dflash-948484b97678`.

SGLang's V100 backend contains dedicated small-Q attention designs using Q16, KV32/KV64, split context, and 80-SM-aware launch geometry. LMDeploy's SM70 attention registry was extended with runtime-selectable Q16/Q32 and KV32 variants for both the draft's 128-dimensional heads and the target verifier's 256-dimensional heads.

The first single-build matrix appeared to show small cycle-cost improvements, but separate-process acceptance varied substantially. A longer five-trial confirmation falsified the apparent gain: Q64/KV64 achieved 60.74 tok/s at commit length 2.669, or 43.94 ms/verification step, while combined draft-Q16/KV32 plus target-Q32/KV32 achieved 57.43 tok/s at commit length 2.537, or 44.17 ms/step. The alternate policy passed audited identity but was 0.5% slower after acceptance normalization. Tile size alone is therefore not the missing optimization; SGLang's grouped heads, split-context scheduling, persistent buffers, and graph capture are the material architectural differences. The extra variants were removed. Artifacts: `/results/20260828_202608-dflash-attention-tile-446144af923c` and `/results/20260828_203751-dflash-attention-tile-confirm-e3aa21413e27`.

## Stable workspace and one-pass rejection

Commit `622e491b` completed a one-build dynamic/workspace/workspace-plus-one-pass comparison on island 2. All arms used five trials. Exact audited K=0/K=7 identity passed for the selected arm, and the first-real-block parity smoke wrote and validated complete manifests on all four TP ranks.

Acceptance varied across fresh processes, so cycle time was normalized by each arm's final commit length. The unprofiled dynamic, workspace, and one-pass arms measured approximately 43.09, 43.05, and 42.67 ms per cycle. Under matched Nsight profiling, all three arms had commit length 2.311 and measured 48.53, 48.08, and 47.17 ms per cycle.

The first persistent workspace slice reduced `cudaMallocFromPoolAsync` and `cudaFreeAsync` calls from 108,040 each to 94,556 each in the captured aggregate. `dflashDraftAndSelect` fell from 7.46 to 7.04 ms average. The direct wall-time change remained small, but stable addresses are required for CUDA graph capture.

The one-pass deterministic top-2 rejection kernel removed the second vocabulary scan. Its profiled kernel average fell from 1.495 to 0.971 ms, a 35% reduction. Relative to the workspace control, profiled normalized cycle time improved by 1.9%; unprofiled normalized cycle time improved by 0.9%. The implementation is retained and enabled by default, with `TM_DFLASH_ONE_PASS_REJECT=0` as the legacy control.

Artifacts: `/results/20260828_230149-dflash-one-pass-reject-622e491b2e4a`.

A second stacked qualification at commit `014fcebd` added phase-owned local/TP top-16 tensors and `UnifiedAttentionLayer` qkv, attention-output, and flattened-KV arenas. Four-rank parity and audited identity passed. The expanded workspace reduced profiled allocator/free calls from 108,040 to 90,476 per captured aggregate (16.3%). Matched commit-length profiling improved normalized cycle time from 48.72 to 48.37 ms (0.7%); five-trial unprofiled normalization improved from 43.26 to 42.57 ms (1.6%). The direct gain is modest, but the DFlash draft path now has stable addresses through attention and candidate selection. Artifacts: `/results/20260828_231817-dflash-one-pass-reject-014fcebdbe49`.

## Selector graph and direct paged Q=8 experiments

Commit `347ac480` fixed TP4 selector-graph capture by changing stream capture from global to thread-local mode. All four ranks then captured and replayed successfully.

The selector-only graph did not reduce the speculative cycle. Its matched profile changed normalized cycle time from 47.20 to 48.17 ms. The `dflashDraftAndSelect` range remained 7.071 versus 7.068 ms. Unprofiled reruns changed sign, so this graph slice is rejected as a speed path. Exact audited identity passed.

Commit `79fc95e7` fixed direct paged Q=8 parameter assembly. The paged path bypassed flattened KV for eligible SM70 proposal attention and reduced allocator/free calls from 90,476 to 86,192. Five-trial normalization changed from 42.66 to 41.61 ms, but matched profiling changed from 47.20 to 47.64 ms.

Repeated exactness controls confirmed that the existing fresh-process oracle is unstable rather than exposing a paged-specific failure. Three flattened controls split at positions 220, 220, and 145. Three paged controls produced one exact pass and two known splits at 220 and 145. There was no unexpected divergence.

First-block comparison was also confounded before the changed attention path: flattened-versus-paged first differed at `target.post_layer_residual` and changed 26 candidate IDs, while flattened-versus-flattened independently first differed at the same boundary and changed 19 candidate IDs. The final selected proposal IDs were exact in both comparisons. Paged Q=8 therefore remains experimental and off by default, but is retained as the current full-graph prerequisite.

Artifacts: `/results/20260828_234307-dflash-graph-paged-79fc95e78c15`, `/results/20260829_000639-dflash-paged-followup-79fc95e78c15`, `/results/20260829_001458-dflash-paged-parity-79fc95e78c15`, and `/results/20260829_001656-dflash-parity-control-79fc95e78c15`.

The larger draft-plus-selector graph behind `TM_DFLASH_DRAFT_GRAPH=1` captured and replayed successfully on all four TP ranks. Audited identity passed. Its five-trial normalized cycle improved only from 41.88 to 41.65 ms, about 0.6%. Matched profiling showed the draft/select range itself falling from 6.99 to 6.60 ms, but whole-cycle normalization regressed from 47.17 to 48.63 ms as target/rollback scheduling shifted under Nsight. CUDA kernel launches fell from 193,044 to 166,932 and 204 graph launches covered the profiled replay cycles. The graph remains off by default and is retained as infrastructure for broader capture, not as a qualified standalone speed default. Artifacts: `/results/20260829_002508-dflash-draft-graph-2892674f32d2`.

A subsequent per-layer target-attention graph prototype captured and replayed all 64 rank/layer instances. In the matched profile both arms had commit length 2.311 over 103 verification steps. The graph reduced `targetVerify` from 14.235 to 13.904 ms and cut kernel launches from 166,932 to 144,084, but it required sixteen extra graph launches per verification. Whole profiled request time regressed from 2.8897 to 2.9129 seconds, about 0.8%. The five-trial raw throughput increase was not attribution-safe because fresh processes followed materially different acceptance trajectories. Audited identity hit only the known position-220 split. This slice was rejected and removed; any future target graph must capture a larger contiguous region and amortize graph-launch overhead. Artifacts: `/results/20260829_010135-dflash-target-attention-graph-a632070eaea9`.

## Current speculative-cycle attribution

A matched K=0/K=7 Nsight Systems run at commit `930baf48` profiled the audited prompt from one wheel. K=7 decoded at 47.59 tok/s under profiler overhead with commit length 2.311, which implies 48.6 ms per verification cycle.

| Host range | Average per rank |
| --- | ---: |
| `targetVerify` | 15.02 ms |
| `speculativeRollback` | 23.54 ms |
| `dflashDraftAndSelect` | 7.40 ms |
| `speculativeReject` | 0.09 ms |

Context-KV work adds approximately 0.52 ms per verification. Together these ranges explain about 96% of normalized cycle time. `speculativeRollback` includes outstanding target GPU work before its first host verdict read, so it is not independent computation.

A subsequent SQLite correlation-ID join connected each NVTX range to CUDA runtime launches and then to GPU kernels. Across 204 verification cycles, `speculativeRollback` itself launched only 0.68 ms total of `AppendTokenIdsKernel`. Work launched in `targetVerify` was dominated by approximately 9.87 ms/cycle of FP8 GEMMs, 5.99 ms/cycle of head-dim-256 attention, 4.19 ms/cycle of CUTLASS GEMMs, 3.86 ms/cycle of chunked GDN, and 2.51 ms/cycle of NCCL kernels. Kernel-duration sums can overlap, but the join proves the long rollback range is chiefly the synchronization point for target GPU work. It also promotes grouped direct-paged target attention over standalone host-publication changes.

CUDA API attribution initially suggested host control as the first target. The K=7 capture issued 25,332 `cudaMemcpyAsync` calls that consumed 6.07 seconds of aggregate API time, or 59.1% of reported CUDA API time. It also issued 108,040 `cudaMallocFromPoolAsync` and 108,040 `cudaFreeAsync` calls. The speculative verdict, length, tip, and candidate readbacks use temporary pageable host buffers. Pageable asynchronous copies can block during host staging before the explicit stream synchronization.

The three-arm result falsified pageable staging as a useful optimization. Acceptance-normalized cycle time was 43.04 ms with pageable buffers, 42.92 ms with pinned buffers, and 42.99 ms with pinned buffers plus a combined rollback barrier. The maximum difference was 0.3%. Exact audited identity passed for the combined arm. Raw decode throughput was not used for attribution because fresh-process commit length differed at 2.620, 2.478, and 2.669.

The large `cudaMemcpyAsync` API attribution therefore mostly represents outstanding GPU work encountered at host-control synchronization boundaries, not removable host staging overhead. The pinned hot-path buffers and controls were removed. The trace smoke subtest requested only eight outputs, which suppresses a seven-token draft near the generation limit, so it produced no parity trace. A later 64-output run validated complete first-block manifests on all four TP ranks. Experiment artifacts: `/results/20260828_214332-dflash-pinned-staging-057db9a76ebc`. Profile artifacts: `/results/20260828_211752-nsys-dflash-930baf48a115`.

## Grouped direct-paged target verification

The SM70 grouped Q8H4 kernel combines all eight verifier query positions with up to four GQA heads in one MMA CTA. The audited local 6Q:1KV shape uses two 4+2 head CTAs and eight context splits at 1K. All 64 TP-rank/full-layer outputs were bit-exact against the direct-paged reference. Audited 256-token generation identity passed with the draft graph both disabled and enabled, and a 64-token-context smoke proved capture on all four ranks.

Five-trial 1K cycle normalization improved from 42.69 ms flattened and 41.63 ms direct-paged to 38.42 ms grouped. Direct-paged and grouped followed the same 2.756 commit-length trajectory, giving an attribution-safe decode increase from 66.20 to 71.73 tok/s. Matched profiles followed the same 2.311 trajectory and improved from 47.33 ms flattened and 46.83 ms direct-paged to 44.04 ms grouped. The head-dim-256 target attention kernel fell from 1,252.7 to 535.6 ms aggregate, a 57.2% reduction.

At 8K, equal commit length 2.750 produced 69.17 versus 44.67 ms normalized cycles. At 25K, equal commit length 3.812 produced 131.27 versus 52.31 ms. This is the first structural attention change to produce a large, context-scaling gain. The grouped path is now default-on, with `TM_DFLASH_GROUPED_PAGED_Q8=0` retaining controls. Artifacts: `/results/20260829_044633-dflash-grouped-paged-q8-82bcb2ed29a4`.

## SM70 block-FP8 M=8 tile sweep

A one-build experiment registered seven additional V100 `Config_E4M3` kernels around the existing 8x128x64 tile. Exhaustive per-device tuning covered CTA-N 64/128/256 and CTA-K 32/64/128. The three block-FP8 M=8 problems actually dispatched during target verification are gate/up `8x8704x5120`, down `8x5120x4352`, and output projection `8x5120x1536`; QKV and lm_head use other kernel families. The tuner selected 8x128x32 for gate/up, 8x128x128 for down, and 8x64x128 or 8x128x128 for output projection. Occupancy evidence showed zero local-memory spill, 79-152 registers, and two to twelve active CTAs.

The tile sweep did not improve execution. Five-trial acceptance-normalized cycle time regressed from 38.87 to 39.35 ms, or 1.25%. Matched profiles had identical commit length 2.311 and changed normalized cycle time only from 44.58 to 44.48 ms, a 0.23% apparent gain within variance. More decisively, same-capture-range M=8 block-FP8 kernel time regressed from 11.10 to 11.52 ms per `targetVerify` NVTX instance, or 3.74%. The tuner microbenchmarks did not predict the aggregate workload result.

The job stopped before candidate identity because its first analyzer incorrectly required three shapes not served by this catalog. The captured performance evidence is sufficient to reject the arm; no candidate kernel is retained or claimed correct. Corrected analysis is stored beside the incomplete run. Future FP8 GEMM work should start with Nsight Compute on the baseline kernel or pursue structural fusion/layout changes, not another tile-only sweep. Artifacts: `/results/20260829_054609-dflash-fp8-m8-tiles-a546ab05061f`.

## SM70 block-FP8 M=8 Nsight Compute characterization

A standalone single-GPU driver now invokes TurboMind's real `Gemm` implementation and production autotuner without bringing NCCL under Nsight Compute replay. DCGM was paused through the cluster's `profiling-paused` node label only while collecting counters, then restored. The production 8x128x64 kernel selected split-K 5/8/8 for gate-up/down/output. Its measured durations were 67.39, 46.98, and 25.57 microseconds respectively.

All three launches are latency- and eligibility-limited rather than DRAM-bandwidth-limited. Compute throughput was 21.38-31.77%, DRAM throughput only 1.51-2.73%, and L2 hit rate 94.31-96.28%. Each scheduler had 3.56-4.10 active warps but only 0.42-0.51 eligible warps, issuing 0.26-0.34 warps per cycle. The kernel uses 94 registers per thread and 18.98 KiB shared memory per CTA; both independently cap residency at five CTAs, giving 31.25% theoretical and 21.79-25.30% achieved occupancy. L1/TEX throughput was much higher at 47.17-73.62%, consistent with the packed E4M3 load, software FP8-to-FP16 conversion, and scale path being the structural target.

This evidence closes blind tile exploration. The next FP8 experiment must reduce conversion/scale dependency work or resource lifetime while preserving the tuned 8x128x64 geometry and split-K choices. Artifacts: `/results/20260829_102845-dflash-fp8-m8-ncu-a546ab05061f`.

## Rollback barrier merge

A five-trial A/B queued rollback verdict, published-length, and tip readbacks behind one barrier instead of two. The change did not improve acceptance-normalized cycle time: the legacy path took 43.11 ms per step and the combined path took 43.13 ms. Raw decode differed because separate processes followed different acceptance trajectories. The combined arm also hit the known audited position-220 near-tie. Since the normalized result showed no gain, the implementation and runtime flag were removed. Artifacts: `/results/20260828_205537-dflash-rollback-sync-e49a2f50daff`.

## Phase-owned sequence frontier

The rollback result now carries each row's exact accepted host frontier into the next draft, removing the redundant pre-draft sequence-length device-to-host copy and synchronization. A legacy-readback control remains behind `TM_DFLASH_LEGACY_FRONTIER_READBACK=1`, and invalid or mixed state falls back safely.

The matched profile had identical commit length 2.311 over 103 verification steps. It removed 236 `cudaMemcpyAsync` calls and shifted work out of the draft/rollback synchronization boundaries, but whole profiled request time changed from 2.9273 to 2.9427 seconds, a 0.5% regression within profile variance. Five unprofiled trials suggested only about a 0.4% acceptance-normalized improvement. No safety fallback occurred. The audited identity arm hit the runtime's known position-145 near-tie and no unexpected split.

This change is retained as device-control infrastructure, not as a standalone speed claim: one mandatory synchronization is gone, but later rollback/publication waits still absorb the outstanding work. Artifacts: `/results/20260829_011308-dflash-frontier-followup-a632070eaea9`.

## Parallel local top-16 reduction

Nsight measured the per-rank `DFlashTopK16Half` candidate scan and serial 256-lane merge at about 1.1 ms per speculative cycle. Replacing the serial shared-memory merge with `cub::BlockReduce<TopK<float,16>>` improved acceptance-normalized cycle time from 43.87 to 43.45 ms, approximately **1.0%**, over five measured trials. The separate-process acceptance trajectories differed, so raw decode was 57.36 versus 55.56 tok/s and is not the attribution metric; both implementations compute the same deterministic score/ID ordering, and the CUB arm passed the audited 256-token identity gate. CUB is now the default, with `TM_DFLASH_CUB_TOPK=0` retaining the legacy control. Artifacts: `/results/20260828_204616-dflash-cub-topk-9ebbe196cada`.

## Acceptance gap

Current audited comparison:

| Runtime | Commit length |
| --- | ---: |
| SGLang DFlash2 | 3.765 |
| LMDeploy DFlash2 | 2.664 |

A corrected same-input TP4 parity capture at commit `b753831d` forced the exact block `[1144, 248070 x 7]` at positions `1000..1007`. All four ranks emitted 102 complete boundaries, and block embeddings were bit-identical. The earliest mismatch is the target residual trajectory, not block construction or feature ordering. RMS error grows across checkpoint target layers `[5, 19, 33, 47, 61]` as `0.1075`, `0.3547`, `0.6800`, `1.3486`, and `4.6666`; the final feature's maximum absolute error is `114.18`. SGLang's native target also chooses token `1596` where TurboMind chooses `1144`, independently confirming materially different target numerics.

Commit `4fe99716` completed a four-rank TurboMind-only context replay isolation. Replaying SGLang's exact FP16 `[1, 25600]` target residual made the captured input bit-identical. The context FC difference collapsed from RMS `44.2053` to `0.05926`; after RMSNorm, output parity passed with maximum absolute error `0.00390625` and RMS `0.000277`. Native context norm RMS was `0.39279`. Thus the context projector is functionally aligned after normalization, while upstream target-model numerical drift is the dominant context mismatch. The first independent draft-side mismatch is now `block.initial_norm` at maximum absolute `0.015625` and RMS `0.000728`.

Artifacts: `/results/20260829_032508-sglang-dflash-parity-b753831db680` and `/results/20260829_035037-dflash-context-replay-4fe9971622bc`.

A follow-up `TM_DFLASH_FULL_PRODUCT_RMSNORM` arm exactly reproduced SGLang's SM70 `block.initial_norm` and brought the next grouped-convolution projection and side-0 output within FP16 parity tolerance. It did not improve acceptance: both five-trial arms had commit length `2.669`. Acceptance-normalized cycle cost changed from `41.76` to `41.98` ms, about 0.5% slower. Audited identity hit the runtime's known unstable position-220 split. The first subsequent material mismatch is now the attention branch before output convolution; under reduce-first parity, layer-0 `attention.conv_side1` differs at max abs `21.06`, RMS `0.2188`. Because the exact norm match produced no acceptance or speed gain, the kernel and flag were removed. Artifacts: `/results/20260829_040158-dflash-full-product-rmsnorm-1dced68ec22b` and `/results/20260829_041033-dflash-reduce-first-parity`.

At LMDeploy's current cycle cost, immediately matching SGLang acceptance would improve throughput materially but still not guarantee 2.2x. Both fidelity and cycle cost need work.

The next acceptance investigation is ordered as follows:

1. move both branch TP reductions before output convolution and W2 row-scale restoration;
2. inspect the checkpoint architecture, unmatched keys, per-layer attention contracts, and RoPE ownership;
3. compare intermediate projected context, draft hidden states, candidates, and selected path against SGLang if the gap remains;
4. only then pursue CUDA graph capture of fixed-shape target verification and draft execution.

## Open audits

Independent read-only audits were launched to examine:

- residual and normalization order;
- grouped-convolution indexing;
- attention position and draft-KV lifecycle after partial acceptance;
- selector lattice semantics;
- target hidden-state selection after verification.

Their findings must be classified as confirmed mismatches or hypotheses before changing runtime behavior.
