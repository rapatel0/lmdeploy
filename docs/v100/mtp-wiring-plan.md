# Wiring MTP so it actually accelerates decoding

## What is wrong today

The current build drafts and throws the drafts away. Measured on four V100s,
tp=4, Qwen3.8-27B-FP8, 1051-token prompt, 256 output tokens:

```
decode tok/s   K=0 55.26   ->   K=4 38.35    0.694x
acceptance     398/992 = 40.1%
```

So drafting costs 31% of throughput and returns nothing. That is not a useful
result. Benchmarking a discard path measured the cost of work that was designed
to be wasted.

The 40.1% acceptance figure is also not trustworthy as a predictor. The MTP KV
slot is never populated, so the draft attends to uninitialised cache entries.
Whatever that number means, it is not "what acceptance would be if this were
wired up".

## Why the drafts are discarded

Three things are missing, and they are missing together.

**1. The engine emits exactly one token per step.** `engine.cc` sets

```cpp
c.inflight_new_tokens = c.generating;   // a bool
```

so a step can advance `seq_len` by at most one. Accepting N drafted tokens
requires the engine to write N+1 tokens and advance `seq_len` by N+1.

**2. There is no verify/reject/rollback machinery.** `BatchOp` has
`kPrepare/kForward/kUnprep` and nothing else. There is no operation that
compares target logits against drafted tokens, and none that rolls back the
sequence when a draft is rejected.

**3. The scheduler reserves no space for drafts.** Blocks are allocated for one
new token per step, so a K+1-token forward has nowhere to put the extra
entries.

## The reference implementation already solves this

`./lmdeploy-v100` is the same upstream base with a working speculative path.
It is the second independent design SPECULATION.md refers to, and it is C++ in
the same tree layout, so it ports directly rather than by analogy.

The pieces, by file:

| Piece | Reference location | Mine |
| --- | --- | --- |
| `kDraft`, `kReject`, `kRollback` ops | `engine/batch.h` | absent |
| 5-step steady-state pipeline | `engine/model_executor.cc:155` `RunWithDrafts` | absent |
| First-decode draft priming | `engine/model_executor.cc:230` `RunDraftOnly` | absent |
| Block reservation for K drafts | `engine/engine.cc:809` alpha/beta | absent |
| Draft token injection into batch | `engine/engine.cc:623` | absent |
| Multi-token `Update` | `engine/engine.cc:769` | one token only |
| `SpeculativeDecode` block_ptrs | `engine/engine.cc:871` | absent |

Line counts show the gap:

```
                         reference   mine
engine/engine.cc              1056    973
engine/model_executor.cc       306    152
models/llama/mtp_predictor.cc  524    421
```

### The steady-state pipeline

```
Forward(K+1 tokens)  ->  Reject  ->  Rollback  ->  Draft
```

Verify-in-next-forward: the K+1 tokens `[bonus, D0..D_{K-1}]` go through one
prefill-shaped forward. The resulting logits are compared against the drafts;
the longest correct prefix is accepted; the sequence rolls back to the first
rejection; then MTP drafts again from the new tip.

The economic claim this rests on -- that one K+1-shaped forward beats K+1
decode forwards -- is the thing `bench_prefill_vs_decode.py` was written to
test, and it is why the win exists at all.

## Constraints that shape the port

**K=1 is the validated configuration in the reference.** Its HEAD commit
(`d7c29f8`) restricts MTD to single-layer MTP explicitly:

> Multi-layer MTP models (e.g. Step3p5 with 3 MTP layers) need exact verifier
> logits from prefill kernel. Only allow MTD for validated single-layer 2-token
> verifier case (Qwen3.5: bonus + 1 draft).

Qwen3.5 is single-layer MTP, so K=1 is the supported path for this checkpoint.
At K=1 the ceiling is 2.0 accepted tokens per step, and the reference measures
~1.4x. That is the honest near-term target.

**2.2x is a K~4 number and needs accept length ~4.2.** SPECULATION.md is
explicit that comparing 1.4x at K=1 against 2.2x at K~4 is not apples to
apples. Reaching the higher figure needs either multi-layer verification (which
the reference disabled as unvalidated) or the DFlash2 five-layer drafter, which
`MTPPredictor` cannot currently load. Neither is a wiring change.

**GDN state is not snapshotted.** The reference documents this deliberately:
on rejection the rejected drafts' GDN contributions stay in the state, which is
a small error at K=1 and does not accumulate. Restoring the previous state
instead loses the bonus token's contribution and degrades output badly. There
is a `TODO` for two-phase forward at K>1.

## Order of work

1. Add `kDraft`, `kReject`, `kRollback` to `BatchOp`.
2. Port block reservation (alpha/beta) so K drafts have somewhere to live.
3. Port draft-token injection into the batch.
4. Port `RunWithDrafts` / `RunDraftOnly` into `model_executor.cc`.
5. Make `Update` accept N+1 tokens instead of one.
6. Seed the MTP KV slot, without which acceptance stays meaningless.

## How this gets verified

The existing gates already cover correctness and stay in force: 15/15 answer
checks per configuration, greedy determinism across rounds, negative controls
that reject wrong answers, and the multi-block prefill case.

Two additions specific to this work:

- **Output must be identical to K=0.** Speculative decoding is an exactness-
  preserving optimisation under greedy sampling. If accepting drafts changes
  the text, the verification is wrong. This is a stronger check than "answers
  are still correct" and it is the one that matters.
- **Accept length must be reported and non-trivial.** A pipeline that accepts
  nothing is the current situation wearing new code. The benchmark already
  fails without an engine-side acceptance line.

Success is `decode tok/s at K=1 > decode tok/s at K=0` on the same job, same
GPUs, same wheel, with byte-identical output. Anything less is not a win.
