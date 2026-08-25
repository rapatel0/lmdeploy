# The SGLang gap: what we actually have to beat

The goal is Qwen3.8 on LMDeploy at V100, at or better than SGLang. This file
records the target numbers and the honest distance to them.

The short version: we are behind, and the campaign plan does not currently
target the thing that would close the gap.

## The target numbers

`sglang-V100/benchmark/qwen36_27b_fp8_v100_20260801/results.csv`, same model
class, same hardware, batch of 1, 1024 in, 256 out:

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

Only one lever remains.

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
