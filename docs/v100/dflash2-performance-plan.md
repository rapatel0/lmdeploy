# DFlash2 V100 performance plan

Status: active
Date: 2026-08-29
Evidence log: [`dflash2-investigation.md`](./dflash2-investigation.md)
Detailed backlog: [`dflash2-todo.md`](./dflash2-todo.md)

## Goal

Close the measured DFlash2 gap against SGLang on four V100 GPUs with TP4 and NCCL.

Keep exact target output identity for every accepted optimization.

## Baseline and target

| Metric | LMDeploy baseline | SGLang reference | Final target |
| --- | ---: | ---: | ---: |
| Commit length | 2.664 | 3.765 | at least 3.5, stretch 3.765 |
| Verification steps/s | 23.0 | 36.3 | at least 33.3, stretch 36.3 |
| Cycle time | 43.4 ms | 27.6 ms | at most 30 ms, stretch 27.6 ms |
| Decode | 61.39 tok/s | 136.6 tok/s | at least 2.2x matched K=0 |

Treat the gap as two independent tracks:

1. Raise commit length by at least 1.31x.
2. Reduce verification cycle time by at least 1.45x.

Run both tracks in parallel after the baseline gate.

## Rules for every experiment

1. Stack related variants into one build behind runtime flags.
2. Run at least five measured trials per arm.
3. Report raw decode and acceptance-normalized cycle time.
4. Use the audited 1,000-token prompt and its fixed SHA-256.
5. Run exact K=0/K=7 identity for every selected arm.
6. Keep TP4 on one NVLink island and verify NCCL transport.
7. Use Nsight Systems before any claim above one percent.
8. Remove variants that fail confirmation.
9. Record every artifact path in the evidence log.

## Execution priority

Use two passes. Finish low-risk changes that need no new attention algorithm, then move directly to structural gains.

### Pass 1: easiest changes

1. Stabilize proposal and selector buffers per phase.
2. Replace the rejection kernel's second vocabulary scan with one deterministic top-2 reduction.
3. Stabilize draft-layer convolution, projection, activation, and residual buffers.
4. Keep only changes that pass identity and either reduce cycle time or provide required graph-safe addresses.

### Pass 2: largest speedups

1. Capture the complete draft and selector path in a CUDA graph.
2. Capture fixed-shape target verification after its addresses are stable.
3. Add a direct paged Q=8 verification attention path that removes KV flattening and exposes more V100 CTAs.
4. Move proposal eligibility, rollback, and publication to device kernels to remove the remaining host boundaries.
5. Use first-block parity to fix the earliest TurboMind draft mismatch and raise commit length toward 3.765.

Do not spend another build on changes whose measured ceiling is below one percent unless they are graph prerequisites.

## Track A: cycle speed

### A0. Freeze the measurement contract — complete

Commit `930baf48` produced matched K=0/K=7 Nsight profiles from one wheel.

The K=7 profile attributed about 96% of the 48.6 ms normalized cycle:

- target verification: 15.02 ms;
- rollback and outstanding GPU tail: 23.54 ms;
- draft and selector: 7.40 ms;
- rejection: 0.09 ms;
- context KV: approximately 0.52 ms per verification.

Artifacts: `/results/20260828_211752-nsys-dflash-930baf48a115`.

### A1. Remove host barriers

- Treat the rollback barrier merge as closed.
- It changed normalized cycle time from 43.11 to 43.13 ms and provided no gain.
- Persistent pinned staging was falsified: pageable, pinned, and pinned-plus-combined arms measured 43.04, 42.92, and 42.99 ms per acceptance-normalized cycle.
- Replace the host sequence-limit check with a device predicate.
- Keep proposal IDs in device memory through verification setup.
- Keep accepted lengths, bonus IDs, and published lengths in device memory.
- Remove host reads from the steady batch-one K=7 cycle.
- Retain a host fallback for EOS, retirement, and request-limit transitions.

Done when the steady cycle contains no mandatory host synchronization.

Target: reduce cycle time from 43.4 ms to at most 39 ms.

### A2. Stabilize all workspace addresses — partial

- Phase-owned workspaces now cover context projection, five draft layers, attention QKV/output/flattened KV, local top-16, TP exchange, selector tensors, and sequence offsets.
- Four-rank parity and audited identity passed for the qualified workspace stack.
- Profiler allocator/free calls fell from 108,040 to 90,476, a 16.3% reduction.
- Matched profiled normalization improved by 0.7%; five-trial unprofiled normalization improved by 1.6%.
- Target verification, rollback, and lower-level library allocations remain.

Done when every graph input and output uses a stable address.

Target: eliminate steady allocation from every captured path.

### A3. Capture the draft and selector graph — capability complete, speed target missed

- Selector-only capture succeeds on all four TP ranks but does not reduce cycle time.
- Full draft-plus-selector capture also succeeds on all ranks and passes audited identity.
- Full capture reduced the draft/select range from 6.99 to 6.60 ms and aggregate kernel launches from 193,044 to 166,932.
- Five-trial whole-cycle normalization improved only 0.6%; matched profiled normalization regressed as target/rollback scheduling shifted.
- Keep both graph modes off by default. Reuse the lifecycle for broader target capture rather than tuning this slice further.

Done: graph capability and NCCL capture are proven. The standalone speed target was not met.

Next target: include enough target verification work to reduce whole-cycle time beyond run variance.

### A4. Capture target verification — per-layer slice rejected

- A per-layer target-attention prototype captured and replayed all 64 TP-rank/layer graphs with stable QKV, paged-attention, gate, and Wo buffers.
- With matched commit length 2.311, `targetVerify` improved from 14.235 to 13.904 ms and kernel launches fell from 166,932 to 144,084.
- Sixteen added graph launches per verification erased that gain: whole profiled request time regressed from 2.8897 to 2.9129 seconds.
- The per-layer branch was removed. Do not revisit this granularity.
- Stabilize the remaining target transformer and recurrent-state addresses, then capture a materially larger contiguous region that amortizes launch overhead.
- Keep commit decisions outside the graph until device publication is complete and preserve fallback for mixed batches and variable verification depths.

Done when broader target verification replays without host intervention and improves matched whole-cycle time.

Target: reduce cycle time to at most 30 ms.

### A5. Add direct paged verification attention — prototype active

- A first SM70 Q=8 block-iterator path now reads paged KV without `invokeFlattenKV_v2_`.
- It reduced allocator/free calls from 90,476 to 86,192.
- Unprofiled normalization improved by 2.5%, but matched profiling regressed by 0.9%.
- This prototype does not yet implement grouped GQA or split-context scheduling.
- Repeated exactness yielded one paged pass and only the known position-145/220 splits; flattened controls hit the same splits.
- Cross-process parity first diverged before attention for both paged and baseline-repeat comparisons, while final selected IDs stayed exact.
- Keep this prototype off by default and retain it only as the current full-graph prerequisite.
- Group GQA query heads and split long contexts only after full-graph qualification proves the prerequisite worthwhile.

Done when the kernel passes identity and beats generic attention at every selected context.

Target: reach 27.6 to 29.0 ms per verification cycle.

### A6. Evaluate target FP8 KV separately

- Add E5M2 target KV as an isolated policy.
- Keep draft KV in FP16.
- Measure conversion cost and bandwidth savings on V100.
- Reject the policy if 1K performance regresses or identity fails.
- Reassess at 8K and 25K contexts.

Do not combine this experiment with attention-kernel attribution.

## Track B: acceptance and fidelity

### B0. Capture first-block parity

- Capture identical target features from LMDeploy and SGLang.
- Compare context FC output and context norm output first.
- Compare every draft layer input and output in execution order.
- Compare convolution, attention, residual norm, and MLP boundaries.
- Compare final hidden states, candidate IDs, unary scores, and selector scores.
- Stop at the first material mismatch.

Done when one earliest mismatch has an exact tensor, layer, and operation boundary.

### B1. Fix the earliest numerical mismatch

- Reproduce only the first mismatch behind one runtime flag.
- Match dtype casts, collective boundaries, scale restoration, and reduction order.
- Run block-level parity before full generation.
- Keep the change only if parity improves without a cycle regression.

Done when the first mismatch matches the agreed tolerance.

### B2. Prove cache and metadata lifecycle correctness

- Poison rejected verifier KV suffixes after every partial acceptance.
- Verify that the next draft block remains unchanged.
- Assert the draft key span after every rollback.
- Rebuild attention metadata after rollback as a control arm.
- Compare candidate blocks and acceptance against the normal path.

Done when rejected state cannot influence the next draft.

### B3. Close remaining layer-level parity gaps

- Repeat first-mismatch localization after each accepted fix.
- Defer grouped-convolution arithmetic until earlier boundaries match.
- Defer RMSNorm reduction parity until the first divergence reaches RMSNorm.
- Keep only changes that improve tensor parity or measured acceptance.

Done when candidate IDs and selector scores match for the audited first block.

### B4. Raise end-to-end acceptance

- Run five trials after every accepted parity fix.
- Report raw and committed acceptance separately.
- Track full accepts, partial accepts, ambiguity events, and EOS events.
- Test the 1K audited prompt and a mixed prompt suite.

Done when commit length reaches at least 3.5 without identity loss.

Stretch target: reach the SGLang reference of 3.765.

## Track C: final qualification

### C0. Run correctness gates

- Run exact K=0/K=7 identity on the audited prompt.
- Run mixed-row retirement and mixed-length batches.
- Run forced rejection, zero acceptance, partial acceptance, and full acceptance.
- Run accepted EOS, verifier-bonus EOS, and request-limit boundaries.
- Run 1K, 8K, and 25K context cases.

Reject the candidate if any required correctness gate fails.

### C1. Run the final performance gate

- Compare K=0 and K=7 from the same build and process policy.
- Run at least five trials per arm.
- Report median and mean decode throughput.
- Report commit length and cycle time independently.
- Report all-GPU utilization, memory, power, NCCL time, and graph launches.

Qualify only if K=7 reaches at least 2.2x matched K=0 throughput.

### C2. Select production defaults

- Remove experimental branches that did not survive confirmation.
- Keep one fallback path for unsupported shapes and graph failures.
- Document all environment controls and their defaults.
- Update the qualification report and active backlog.

## Dependency order

```text
A0 ──> A1 ──> A2 ──> A3 ──> A4 ──> A5
             └──────────────────────> A6

A0 ──> B0 ──> B1 ──> B2 ──> B3 ──> B4

A5 + B4 ──> C0 ──> C1 ──> C2
```

A1 and B0 can proceed in parallel.

A6 remains optional until the graph and attention work reaches its measured limits.

## Immediate execution queue

1. Finish direct paged Q=8 exactness and first-block parity controls.
2. Run the full draft-plus-selector graph on TP4.
3. Run the read-only SGLang parity harness.
4. Fix the earliest cross-runtime acceptance mismatch.
5. Move proposal eligibility, rollback, and publication to device control.
6. Stabilize target-verification addresses.
7. Capture fixed-shape target verification.

## Stop conditions

Stop a cycle-speed branch after two confirmed trials show less than one percent normalized improvement.

Stop a fidelity branch if tensor parity improves but acceptance and identity do not improve.

Do not adopt approximate acceptance or relaxed output identity without explicit approval.
