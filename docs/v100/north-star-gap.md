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

This is **not** a like-for-like comparison with SGLang's 53.35. Two conditions
differ, and both are recorded rather than smoothed over:

- **tp.** Ours is TP2, SGLang's row is TP4. Fewer ranks means less parallelism
  per token, so this understates what the same code does at TP4.
- **Prompt length.** SGLang used 1024 input tokens; this used roughly 20.
  Prefill is excluded from decode tok/s, but KV length affects per-step
  attention cost, so a longer prompt would if anything be slower.

A TP4 rerun at 1024 input tokens is required before any multiplier against
SGLang is quoted. The honest statement today is that our verified batch-1
decode is 29.25 tok/s at TP2 with a short prompt.

## The gap is two independent problems

| Step | tok/s | Multiplier | Cause |
| --- | ---: | ---: | --- |
| Ours, measured, TP2, short prompt | 29.25 | | not yet comparable |
| SGLang single-request, TP4, 1024 in | 53.4 | 1.82x over ours | scheduling and per-request efficiency |
| SGLang speculative | 117.6 | 2.20x over its own baseline | draft-and-verify, accept about 4.2 |

The 1.82x is an upper bound on the first gap, not a measurement of it, because
the conditions differ as described above. The 2.20x speculative multiplier is
SGLang's own baseline against its own speculative run, so it is a clean figure.

Speculative decoding is the more visible lever, but it may be the smaller of
the two multipliers. The first gap has still never been measured under matched
conditions.

Both must close to reach parity.

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
