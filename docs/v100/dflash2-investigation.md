# DFlash2 V100 investigation

Status: active investigation  
Last updated: 2026-08-28  
Target: Qwen3.8-27B-FP8, TP4 V100, DFlash2 block size 8

Action tracker: [`dflash2-todo.md`](./dflash2-todo.md)

## Qualification target

Match the audited SGLang workload and approach its measured speculative result:

- Exact prompt: 1,000 token IDs, SHA-256 `9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01`
- SGLang target-only reference: 58.21 decode tok/s
- SGLang DFlash2 reference: 136.6 decode tok/s
- SGLang average commit length: 3.765 tokens per verification
- Desired uplift: approximately 2.2x or better

LMDeploy must also preserve exact target-only output identity.

## Current result

Latest audited LMDeploy result:

- DFlash2 decode: 42.27 tok/s
- Average committed length: 1.862
- Exact short-workload K=0/K=7 identity: pass
- Original audited DFlash2 result before the selector/context optimizations: 36.13 tok/s
- Cumulative runtime improvement: about 17.0%

This remains unqualified. Acceptance is less than half of SGLang's, and cycle cost is also too high.

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

All four arms passed the existing 256-token short-prompt K=0/K=7 identity gate. Exact audited-prompt identity remains a separate required gate.

## Confirmed TP4 draft-network mismatch

### Collective boundary is on the wrong side of output convolution

SGLang performs, for attention:

`local Wo -> FP16 TP all-reduce -> output grouped convolution -> residual norm`

LMDeploy performs:

`local Wo -> output grouped convolution -> FP16 TP all-reduce inside residual norm`

For MLP, SGLang likewise reduces raw W2 first, then restores dynamic row scale and applies the output convolution. LMDeploy restores the scale and convolves each rank's partial output before reduction.

These forms commute in exact arithmetic but not across FP16 stores and FP16 NCCL. The mismatch occurs at ten branch boundaries across five layers and is now the strongest confirmed draft-fidelity defect.

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

A zero margin can still flag exact ties, as seen in the round-on arm. Permanently changing the default requires exact audited-prompt identity, not only the existing short-prompt gate.

## Acceptance gap

Current audited comparison:

| Runtime | Commit length |
| --- | ---: |
| SGLang DFlash2 | 3.765 |
| LMDeploy DFlash2 | 1.862 |

At LMDeploy's current cycle cost, immediately matching SGLang acceptance would improve throughput materially but still not guarantee 2.2x. Both fidelity and cycle cost need work.

The next acceptance investigation is ordered as follows:

1. run exact audited-prompt K=0/K=7 identity for no context round and zero ambiguity margin;
2. move both branch TP reductions before output convolution and W2 row-scale restoration;
3. inspect the checkpoint architecture, unmatched keys, per-layer attention contracts, and RoPE ownership;
4. compare intermediate projected context, draft hidden states, candidates, and selected path against SGLang if the gap remains;
5. only then pursue CUDA graph capture of fixed-shape target verification and draft execution.

## Open audits

Independent read-only audits were launched to examine:

- residual and normalization order;
- grouped-convolution indexing;
- attention position and draft-KV lifecycle after partial acceptance;
- selector lattice semantics;
- target hidden-state selection after verification.

Their findings must be classified as confirmed mismatches or hypotheses before changing runtime behavior.
