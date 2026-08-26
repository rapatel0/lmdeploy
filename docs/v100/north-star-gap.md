# The SGLang gap: what we actually have to beat

The goal is Qwen3.8 on LMDeploy at V100, at or better than SGLang. This file
records the target numbers and the honest distance to them.

The short version: we are behind, and the campaign plan does not currently
target the thing that would close the gap.

## Correction: the reference baseline was the wrong model

> **This section's original target numbers came from `qwen36_27b_fp8`, which is
> Qwen3.**6**-27B, not our model.** The correct reference for Qwen3.**8**-27B-FP8
> is `benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/`, an audited
> target-only sweep on the same checkpoint we run. Its 1K decode is **58.21
> tok/s**, not 51.73. Every ratio computed from the 3.6 numbers understated the
> gap. The corrected comparison is in "The corrected Qwen3.8 baseline" below.

## The target numbers

`sglang-V100/benchmark/qwen36_27b_fp8_v100_20260801/results.csv`, **Qwen3.6**,
not our model, retained because the campaign reasoned from it for several
phases:

| Mode | KV | Output tok/s | TPOT ms | Accept length |
| --- | --- | ---: | ---: | ---: |
| `target_only` | FP8 | 51.73 | 17.96 | |
| `target_only` | FP16 | 53.35 | 17.48 | |
| `dflash16` | FP8 | 84.48 | 10.41 | 4.23 |
| `dflash16` | FP16 | 117.58 | 7.12 | 4.18 |

`dflash16` is speculative decoding. The accept length near 4.2 means about four
draft tokens are accepted per verify step, which is what turns 53 into 117.

## Our number is not comparable, and it is worse

> **Superseded in part.** The 208.1 tok/s figure below is void: it was measured
> on a wheel built before the `f7b18471` FP8 scale cast, so it timed the
> pre-cast path. The derived 26.0 tok/s per-request figure inherits that fault
> and is also void. A direct batch-1 measurement now exists; see "The measured
> batch-1 number" below. The reasoning about aggregate-versus-per-request
> comparison stands and is the reason this section was written.

Our recorded result is 208.1 tok/s for Qwen3.8-27B-FP8 at TP4. That looks like
a win against 117.58 until the conditions are read.

Ours is aggregate throughput across a batch of 8. SGLang's is a single
request. Per request we are at 26.0 tok/s.

| Measure | Value |
| --- | ---: |
| Ours, aggregate over batch 8 | 208.1 tok/s, VOID |
| Ours, per request, derived by division | 26.0 tok/s, VOID |
| SGLang, no speculation | 53.35 tok/s |
| SGLang, speculative | 117.58 tok/s |

Do not repeat the 208.1 number as a comparison against SGLang. It is an
aggregate against a single-request measurement, and quoting it that way is how
a campaign convinces itself it is winning while losing.

## The measured batch-1 number

Measured directly rather than derived by division, on the wheel verified to
contain the FP8 cast, with every trial gated on non-degenerate output.

| Field | Value |
| --- | --- |
| Wheel | built 19:32, `scale_dtype` present in the packaged source |
| Image | `lmdeploy-v100-base:v2` |
| Node source | `fe2e7eff4989`, `phase/fp8-verified-toolchain` |
| Model | Qwen3.8-27B-FP8, `model_format='fp8'` |
| tp | 2 |
| Prompt | short, roughly 20 tokens, not SGLang's 1024 |
| Output | 256 tokens, greedy |

| Trial | tok/s | TPOT ms | degenerate |
| --- | ---: | ---: | --- |
| 1 | 29.24 | 34.20 | no |
| 2 | 29.28 | 34.15 | no |
| 3 | 29.24 | 34.20 | no |

Mean 29.25 tok/s, TPOT 34.19 ms, spread 0.14 percent. Output was coherent
technical prose, so the number describes real work.

### What it can and cannot be compared to

This is **not** a like-for-like comparison with SGLang's 53.35: it is TP2
against their TP4, and a roughly 20-token prompt against their 1024. It is
kept as the TP2 data point. The matched run is below.

## The matched comparison, TP4 at 1024 input tokens

SGLang's exact conditions: batch 1, 1024 input tokens, 256 output tokens. The
prompt was built by appending text and truncating with the model's own
tokenizer, and the realized input length was asserted at 1024 rather than
assumed. Run on island 2 through `tools/v100/run_island2.sh`, whose guard
confirms four idle GPUs and no island-1 UUID before allocating.

| Trial | tok/s | TPOT ms | degenerate |
| --- | ---: | ---: | --- |
| 1 | 51.06 | 19.58 | no |
| 2 | 50.85 | 19.66 | no |
| 3 | 51.10 | 19.57 | no |

Mean **51.01 tok/s**, TPOT **19.61 ms**, spread 0.49 percent, output coherent.

| Comparison | tok/s | Ratio to ours |
| --- | ---: | ---: |
| Ours, TP4, 1024 in | 51.01 | |
| SGLang `target_only`, FP8 KV | 51.73 | 1.01x |
| SGLang `target_only`, FP16 KV | 53.35 | 1.05x |
| SGLang `dflash16`, FP16 KV | 117.58 | 2.31x |

### This overturns the campaign's stated premise

The section above claimed we were 2.05x behind on raw per-request decode and
that this was "the gap we have never measured". Measured, it is **4.4 percent**
against SGLang's FP16-KV baseline and **1.4 percent** against its FP8-KV
baseline, which is the closer match to our FP8 weights.

The 2.05x was an artifact of dividing a batch-8 aggregate by 8, on a wheel that
predated the FP8 cast fix. Two errors compounded: assuming linear batch scaling,
and timing a broken path. Neither survives direct measurement.

So there is effectively **one gap, not two**. Per-request decode is at parity.
The entire remaining deficit is speculative decoding, worth 2.31x here and
2.20x on SGLang's own baseline-to-speculative comparison.

TPOT tells the same story: 19.61 ms against SGLang's 17.48 ms, a 1.12x
difference, consistent with the throughput ratio and far from 2x.

### What this changes

- **Do not plan work against a 2.05x per-request scheduling gap.** It is not
  there. Any campaign phase justified by closing it needs re-justification.
- **Speculative decoding is the only lever that reaches the north star.** It is
  no longer the smaller of two multipliers; it is the whole remaining gap.
- The MTP finding recorded in `mtp-report.md` therefore matters more, not less:
  LMDeploy already ships six speculative proposers on the PyTorch backend,
  while this campaign runs TurboMind, which has no speculative path. The
  question is which backend reaches the north star, and that is a measurement.

## The gap is two independent problems

| Step | tok/s | Multiplier | Cause |
| --- | ---: | ---: | --- |
| Ours, measured, TP4, 1024 in | 51.01 | | |
| SGLang single-request, TP4, 1024 in | 53.35 | 1.05x over ours | at parity, not a real gap |
| SGLang speculative | 117.58 | 2.20x over its own baseline | draft-and-verify, accept about 4.2 |

Under matched conditions the first gap is 1.05x, which is not a gap worth a
campaign phase. The 2.20x speculative multiplier is SGLang's own baseline
against its own speculative run and remains the clean figure.

## The corrected Qwen3.8 baseline

The numbers above compare against Qwen3.6. Our model is Qwen3.8-27B-FP8, and
`sglang-V100` benchmarks it directly in
`benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/`.

That run is target-only with speculative decoding absent from both the
environment and the server arguments, TP4, FP16 activations, E5M2 KV,
`--attention-backend tilelang_fa_v100`, one request at a time, 256 generated
tokens, cold cache enforced by a warm-then-flush protocol:

| Input | TTFT | Prefill | TPOT | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 342.3 ms | 2,991.9 tok/s | 17.18 ms | **58.21 tok/s** |
| 4,096 | 990.2 ms | 4,136.5 tok/s | 17.36 ms | 57.62 tok/s |
| 25,000 | 6,731.9 ms | 3,713.7 tok/s | 19.70 ms | 50.75 tok/s |
| 70,000 | 23,488.2 ms | 2,980.2 tok/s | 25.76 ms | 38.82 tok/s |
| 128,000 | 54,322.9 ms | 2,356.3 tok/s | 33.29 ms | 30.04 tok/s |

Against our measured 51.01 tok/s at TP4, 1024 in:

| | Decode tok/s | TPOT ms | Ratio to ours |
| --- | ---: | ---: | ---: |
| Ours, TP4, 1024 in | 51.01 | 19.61 | |
| SGLang Qwen3.8 target-only, 1K | 58.21 | 17.18 | **1.14x** |
| SGLang Qwen3.6 target-only FP8 (wrong model) | 51.73 | 17.96 | 1.01x |

**The non-speculative gap is 1.14x, not 1.05x.** It is still not a 2x gap, and
speculation remains the dominant lever, but the earlier "at parity" claim was
too generous and rested on the wrong checkpoint.

Three things this changes:

1. **There is a real per-request deficit**, roughly 14 percent, worth
   attributing before it is dismissed. TPOT shows the same thing: 19.61 ms
   against 17.18 ms.
2. **Their configuration differs from ours in ways we have not matched.** They
   run E5M2 KV and `tilelang_fa_v100`; our 51.01 run used the default
   attention path and FP16 KV. Part of the 14 percent may be configuration
   rather than engine.
3. **The speculative target rises.** DFlash2-8 on the same checkpoint reaches
   136.6 tok/s at 1K, which is 2.35x over their own 58.21 baseline and 2.68x
   over our 51.01.

Only one lever dominates, but the baseline lever is not yet closed.

## Confirming run against the corrected reference

Re-measured on island 2 with `tools/v100/bench_decode.py`, job `bench-tp4-fp8`,
after the reference correction above. Island 1 was running SGLang on all four
of its GPUs and was not touched; the island guard verified no island-1 UUID was
visible before allocating.

| Field | Value |
| --- | --- |
| Wheel | `lmdeploy-0.16.0-cp312-cp312-linux_x86_64.whl` |
| Image | `lmdeploy-v100-base:v2` |
| Model | Qwen3.8-27B-FP8, `model_format='fp8'` |
| tp | 4 |
| session_len | 16384 |
| Prompt | 1,051 encoded tokens, against the reference's 1,024 |
| Output | 256 tokens, greedy, `ignore_eos`, `min_new_tokens == max_new_tokens` |

| Trial | Elapsed s | Output tokens | tok/s | Degenerate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 5.167 | 256 | 49.54 | no |
| 2 | 5.020 | 256 | 50.99 | no |
| 3 | 5.019 | 256 | 51.01 | no |

Mean **50.51 tok/s**. This reproduces the earlier 51.01 figure independently,
with a different driver, so the baseline is solid.

| | Decode tok/s | Ratio |
| --- | ---: | ---: |
| Ours, TP4 FP8, 1,051 in | 50.51 | |
| SGLang Qwen3.8 target-only, 1,024 in | 58.21 | **1.15x** |

Two caveats that make this conservative rather than flattering:

1. Our rate includes the first generated token. The reference derives decode
   from TPOT, which excludes it. Removing it would raise our number slightly.
2. Our prompt is 1,051 tokens against their 1,024, a 2.6 percent longer
   prefill.

Neither closes a 15 percent gap, so the deficit is real.

### The unexplained variables

The reference run differs from ours in three ways that are not yet attributed:

| | Reference | Ours |
| --- | --- | --- |
| KV cache | E5M2 | FP16 |
| Attention backend | `tilelang_fa_v100` | TurboMind default |
| GDN | TileLang prefill, Triton decode | TurboMind |

Attribution comes before optimization. The gap may be configuration rather
than engine, and a phase justified by "our engine is slower" would be premature
until these are matched or ruled out.

### Prefill, and the corrected decode number

The first confirming run measured only an inclusive token rate, so it could not
be compared against the reference on prefill and its decode number carried the
first generated token. A second run measures both.

TTFT comes from a separate `max_new_tokens=1` request, because the synchronous
API returns only on completion. That request is prefill plus one decode step,
so it overstates TTFT and understates prefill throughput.

| Metric | Ours | SGLang, 1,024 in | Ratio |
| --- | ---: | ---: | ---: |
| TTFT | 403.7 ms | 342.3 ms | 1.18x |
| **Prefill** | **2,603.6 tok/s** | **2,991.9 tok/s** | **1.15x** |
| Decode, first token excluded | 55.06 tok/s | 58.21 tok/s | **1.06x** |
| Decode, inclusive | 50.85 tok/s | | |

Two corrections to the earlier entry:

1. **Decode is 1.06x behind, not 1.15x.** Measuring on the reference's own
   definition, excluding the first generated token, closes most of the gap.
   The 1.15x figure compared an inclusive rate against a TPOT-derived one.
2. **Prefill is 1.15x behind**, and that is now the larger of the two gaps.

### What 1Cat-vLLM 1.3.0 did on prefill

`1Cat-vLLM/RELEASE.md`, version 1.3.0. The release reports latencies rather
than a prefill token rate, so the rates below are derived from the 64K shape:

| Matched path | Before | 1.3.0 | Change | Derived tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP16 paged-prefill operator, 64K | 43.605 ms | 31.717 ms | -27.26% | operator only |
| 27B-AWQ TP4 full-model prefill, 64K | 47.98 s | 33.10 s | -31.01% | 1,366 -> 1,980 |
| 27B-AWQ TP2 FP8-KV prefill, 64K | 127.84 s | 62.96 s | -50.75% | 513 -> 1,041 |

The techniques, from the release notes:

- Replaced the D=256 paged-prefill BM16 path with **exact BM32 phase reuse**.
- **All-P scheduling** and conflict-reduced pair scratch.
- **8,096-token chunking**.
- One-pass **FP8 E5M2-to-FP16 KV bridging** and graph-safe FP8 XQA decode.
- Fixed stale CUDA Graph partition metadata so decode scans the live KV length.

They preserved softmax, FP16 probability rounding, FP32 accumulation order and
exact output on the accepted prefill paths, so this is a scheduling and tiling
change rather than a precision trade.

**The roughly 4K tok/s figure is SGLang's, not 1Cat's.** SGLang's audited sweep
on our exact checkpoint reaches **4,136.5 tok/s at 4,096 input**, its peak.
1Cat's 1.3.0 prefill work lands near 1,980 tok/s at 64K on a different model
and weight route, which is a different shape and not directly comparable.

So there are two distinct prefill targets:

| Target | Shape | tok/s |
| --- | --- | ---: |
| SGLang peak, same checkpoint | 4,096 in | 4,136.5 |
| SGLang at our measured shape | 1,024 in | 2,991.9 |
| Ours, measured | 1,051 in | 2,603.6 |

Our 2,603.6 is measured only at roughly 1K. The reference curve rises to its
peak at 4K and falls thereafter, so our shape is not where either project's
prefill is strongest, and a single-point comparison understates the work
remaining at long context. A prefill sweep is the missing measurement.

### Decision: prefill is close enough for now

Prefill is **deferred, not closed**. At 1.15x behind on a single short shape it
is near enough that the campaign should not spend a phase on it while the
speculative multiplier is untouched.

What is parked, and what would restart it:

| Item | State |
| --- | --- |
| Measured points | one, at 1,051 input |
| Missing | a sweep at 4K, 25K, 70K, 128K |
| Known risk | the gap may widen at long context, where the reference itself falls from 4,136 to 2,356 tok/s |
| Available technique | 1Cat-vLLM 1.3.0's BM32 phase reuse, all-P scheduling, 8,096-token chunking, one-pass E5M2-to-FP16 KV bridging |
| Why it is portable in principle | that release preserved softmax, FP16 probability rounding, FP32 accumulation order and exact output, so it is a scheduling and tiling change rather than a precision trade |

Restart this work if a prefill sweep shows the gap widening beyond roughly 1.2x
at long context, or once speculative decoding lands and prefill becomes the
dominant remaining cost.

### Is 1Cat's D=256 paged-prefill work worth porting? No

Assessed rather than assumed. Four findings, any one of which is decisive.

**1. We already have the kernel, and it is not the one they fixed.**
`src/turbomind/kernels/attention/kernel/attention_sm70_256.cu` exists and is
registered for SM70 at D=256:

```cpp
constexpr int kHeadDim = 256;
constexpr int kCTA_Q   = 64;   // their BM16 -> BM32 change targets this
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 16;
```

Their change raises `BLOCK_M` from 16 to 32 on the D=256 low-SMEM path:

```cpp
#define BLOCK_M_256           32
#define BLOCK_M_256_LOW_SMEM  16
static constexpr int BLOCK_M = ... (LOW_SMEM ? (LOW_SMEM_BM32 ? BLOCK_M_256
                                             : BLOCK_M_256_LOW_SMEM) : BLOCK_M_256);
```

Our Q tile is already **64**, twice their improved value. The specific
inefficiency they removed does not exist in our kernel.

**2. The architectural ceiling is about 2 to 3 percent.** Our model is 64
layers, of which only **16 are full attention**; the other 48 are linear
attention and never touch this kernel. Their measured -27.26 percent is on the
prefill operator alone, not the model:

| If attention were this share of prefill time | Max model speedup |
| --- | ---: |
| 50 percent | 1.035x |
| 30 percent | 1.021x |

At our measured 1K shape, GEMM dominates prefill and attention is a small
share, so the realistic figure sits at the low end. Closing a 1.15x gap needs
15 percent, not 2 to 3 percent.

**3. The stacks are not compatible.** Their work lives in a standalone
`flash-attention-v100` tree of 14 CUDA files with its own traits, paged layout,
FP8 KV bridge and decode planner. Ours is TurboMind's `attention` kernels built
around `Impl<MMA_884, ...>`, `AttentionUniversal` and a `Registrar`. This is
not a patch transplant; it is adopting a second attention stack.

**4. Their headline gains are at 64K, not our shape.** The -31.01 percent
full-model prefill result is a 64K measurement on Qwen3.6-27B-AWQ, a different
model and weight route. We have no 64K measurement at all.

**What would change this answer:** a prefill sweep showing our gap widening
sharply at long context. At 64K and beyond attention's share of prefill grows,
and the 16 full-attention layers matter more. Until that sweep exists, porting
this is speculative work against an unmeasured problem.

The transferable idea, if the sweep ever justifies it, is **8,096-token
chunking** and their scheduling changes, not the BM32 tile fix. Those are
engine-level and shape-independent.

### A separate finding from the same run

The staged wheel predates `c9ff318d`:

```text
scale_dtype present   : True
check_scale_range     : False
```

The FP8 cast from `f7b18471` is present, so this measurement is valid. The new
scale-range guard is not in the wheel, so it did not run and could not have
affected the number. Any future run that needs the guard requires a rebuild.

## What this means for the campaign plan

The plan's Phase 5 deliverable is block-scaled INT8 KV: half the KV bytes of
FP16 at FP16 quality. That is a memory-capacity win, not a decode-speed win.

The SGLang data shows KV format is nearly irrelevant to single-request decode
speed on this hardware. FP8 KV against FP16 KV is 51.73 against 53.35 without
speculation, which is under 4 percent, and FP8 KV is the slower of the two.

So the campaign's headline deliverable does not move the north-star metric.
It buys KV capacity, which raises the batch size we can serve, which helps
aggregate throughput. It does not make a single request faster.

This is not an argument to abandon Phase 5. It is an argument to stop treating
it as the thing that closes the SGLang gap, because it is not.

## Two donor assets that do target the gap

**Speculative decoding.** The plan audits `MTPPredictor` in Phase 2 and
explicitly forbids implementing it, deferring it to a separate campaign. That
audit found the donor supplies draft generation only, with no rejection path.
SGLang's `dflash16` is a complete draft-and-verify implementation with a
measured accept length, in `python/sglang/srt/speculative/`. It is the single
largest measured lever in the data we have.

**A V100-specific attention backend.** `flash_attn_v100_backend.py`, 1187
lines, with `forward_decode` and `forward_extend` written for this hardware.
Our Phase 4 task is to select an SM70 attention kernel. This is a working
reference for that decision rather than a starting point to copy.

## Next measurement, before any more porting

We do not have a per-request LMDeploy number measured the way SGLang measures.
The 26.0 tok/s figure above is derived by dividing an aggregate by 8, which
assumes perfect batch scaling and is therefore optimistic.

Measure LMDeploy at batch 1, 1024 in, 256 out, on island 2, and record output
tok/s and TPOT. Until that exists, every comparison here rests on a division.
