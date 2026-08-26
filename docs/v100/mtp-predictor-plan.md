# Building MTPPredictor on TurboMind v0.16.0

## Is MTP working today? No

The weights load. Nothing drafts. Three facts settle it:

1. Every `[MTP]` line in every run log is one of the two records the loader
   itself emits. A search for `draft`, `accept`, `propose` or `speculative`
   across both run logs returns only those lines echoed back.
2. Nothing reads `ModelWeight.mtp` after loading. A search across
   `src/turbomind` for `mtp` outside `mtp_weight.{h,cc}` returns three
   comments and no code.
3. `MTPPredictor` does not exist in this tree, and neither does any draft or
   verify path.

The measured 51.01 tok/s is therefore a target-only number. It shows that
carrying the extra weights is free. It is not a speculation result.

## The reference does not port

`lmdeploy-v100` has a working `MTPPredictor`, 524 lines plus a 107-line
header. It cannot be copied, because the layer it depends on changed shape
between versions.

| | fork, v0.12.0 | ours, v0.16.0 |
| --- | --- | --- |
| `UnifiedAttentionLayer` ctor | `(ModelParam, AttentionParam, EngineParam, tp_size, Context, phases, init)` | `(vector<AttentionWeight*>, CacheRegistry&, EngineParam, Context, phases)` |
| KV slot selection | `mtp_attn_layer_offset_`, computed by counting full-attention layers | `CacheRegistry`, which owns allocation |
| Weight access | `LlamaWeight` with raw `Tensor` members | typed X-macro modules |

The fork picks its KV layer index arithmetically:

```cpp
mtp_kv_layer_idx = mtp_attn_layer_offset_ + mtp_layer_idx;
```

That works only because v0.12.0 indexes the KV cache by layer number. Our
v0.16.0 routes every attention layer through `CacheRegistry`, which hands out
slots. So the central question for MTP is not how to run the layer. It is how
the MTP layer obtains a KV slot that the target's 16 full-attention layers do
not already own.

## What the draft step needs

Per drafted token, from the reference's `ForwardStep`:

```
embed(last_token)                     shared embed_tokens
  -> pre_fc_norm_embedding
hidden_state from the target
  -> pre_fc_norm_hidden
concat both -> [batch, 2*hidden]
  -> fc                               [5120, 10240], BF16
  -> attention                        full attention, needs a KV slot
  -> ffn or moe
  -> final_norm
  -> lm_head                          shared with the target
  -> argmax                           one draft token
```

For a depth of four, run that five-stage chain four times, feeding each
drafted token back as `last_token`. Then the target verifies all four in one
forward pass.

## Work items, in dependency order

1. **KV slot for the MTP layer.** Register one extra attention layer with
   `CacheRegistry` so the draft attention has somewhere to write. Without
   this nothing else can run. This is the item with no reference to copy.
2. **`MTPPredictor` class.** Construct a `UnifiedAttentionLayer` from the
   loaded `MTPLayerWeight`, plus an `FfnWeight` or `MoeWeight` path, and
   implement `ForwardStep` as above.
3. **Draft loop.** Repeat `ForwardStep` K times, carrying the hidden state
   and the last drafted token.
4. **Verification and rollback.** The target scores the K drafts in one pass.
   Accept the longest matching prefix, then roll back the KV cache and the
   GDN recurrent state for every rejected position.
5. **Accept-length measurement.** The only correctness test that works.

## Why step 4 is the hard one, and step 5 is mandatory

This model has 48 linear-attention layers alongside its 16 full-attention
layers. Full attention rolls back by discarding KV entries. Gated DeltaNet
carries a recurrent state that a rejected token has already mutated, so
rejecting a draft requires restoring that state. A wrong rollback does not
crash and does not corrupt text: the target re-generates the correct token
anyway. It shows up only as drafts that are never accepted.

That is why accept length is the correctness test. **Wrong speculation is
slow, not wrong.** A build that drafts four tokens and accepts zero looks
identical in output to one that works, and reports a lower token rate than
no speculation at all. Any claim that MTP works must cite an accept length
above zero, measured, not inferred.

## Verification: what the scheduler makes hard

Verifying K drafts needs the target to score K+1 positions in one forward, so
the scheduler must admit K+1 tokens for a generating sequence instead of one.
That length is not a constant to change. It is derived:

```cpp
// scheduler.cc
const int begin   = s.resume_len + s.inflight_input_len;
const int ctx_end = s.seq_len + s.inflight_new_tokens;
const int end     = ClampForwardEnd(s, begin, begin + admitted, ctx_end);
s.input_len       = end - begin;
```

So `input_len` falls out of `admitted`, the block-size alignment inside
`ClampForwardEnd`, and the checkpoint interval. Three consequences worth
stating before writing any code:

1. **The drafted tokens have no sequence positions yet.** They are candidates,
   not history. Admitting K+1 means reserving KV for positions that may be
   rejected, so the block accounting has to tolerate a forward whose accepted
   length is shorter than its admitted length.
2. **`ClampForwardEnd` aligns down to a block boundary** when `desired <
   ctx_end`. A K+1 decode step can therefore be clipped back to a shorter
   length, silently, which would look like drafts being rejected.
3. **Checkpoint publication is position-driven.** Publishing a checkpoint at a
   position that later gets rolled back would poison the prefix cache for
   every other request that reuses it. This is the failure with the widest
   blast radius, because it escapes the speculating request.

None of this makes the work impossible. It does mean verification is a
scheduler change, not a model change, and it is larger than the predictor
itself.

## Status: the draft loop runs (2026-08-26, commit 8814428e)

`VERIFY_MTP_DRAFT_PASS` on 4x V100, tp=4, Qwen3.8-27B-FP8. Artifacts persisted
to the node at `/localpool/lmdeploy-v100-next/results/`.

```
[MTP] drafted 4 token(s) for 1 sequence(s); drafts are not yet verified   (x4 ranks)
  identical, as required while drafts are discarded
```

What this proves: the predictor constructs, projects, runs K decode steps
through attention and FFN, and produces argmax tokens on all four ranks without
aborting -- and the target's output is byte-identical to the baseline, which is
the only correctness claim available while drafts are discarded.

What it does NOT prove, and must not be read as: that the drafts are *good*, or
that any of them would be accepted. Per the section below, the draft attends to
uninitialised KV at a position that never advances. `drafted 4 token(s)` counts
tokens *emitted*, not tokens worth emitting. The identical-output check passes
trivially here precisely because nothing consumes the drafts -- it would pass
equally well if the predictor emitted four random ids.

So this closes the "does the draft path execute" question and nothing more. The
first number that means anything is accept length, and that is not measurable
until the KV seeding below is done.

## The economics work: 51.5x, and robust to being badly wrong

Measured on 4x V100, tp=4, Qwen3.8-27B-FP8 (commit 488ca359):

```
prefill:    328.9 us/token   (marginal, 512 extra prompt tokens)
decode :  16936.4 us/token   (marginal, 64 extra generated tokens)
ratio:       51.5x
```

Decode at 16.9 ms/step is 59 tok/s. Sanity check: 27B weights over 4 ranks is
~6.8 GB/GPU, and at ~900 GB/s HBM that floors a step at ~7.5 ms. Measured is
~2.2x that floor, which is what one expects from tp=4 with a per-layer NCCL
all-reduce over PCIe rather than NVLink. The number is plausible, not an
artefact.

Both costs are differenced, so model load, request overhead and the single
decode step cancel. The one known bias is in the decode figure: KV grows across
the 64 extra steps, so later steps carry slightly more attention work. At
`seq_len` 40-104 that is negligible against a 27B weight read, and it biases
decode *upward*, i.e. against the conclusion.

Implied ceiling at the measured accept lengths:

```
K=1, accepted 0.64  ->  1.61x
K=4, accepted 0.87  ->  1.74x
```

**Why this settles the question rather than merely supporting it:** the
conclusion holds even if the ratio is wrong by an order of magnitude.

```
ratio 51.5x  ->  K=1 1.61x   K=4 1.74x
ratio 25.0x  ->  K=1 1.58x   K=4 1.61x
ratio 10.0x  ->  K=1 1.49x   K=4 1.34x
ratio  5.0x  ->  K=1 1.37x   K=4 1.04x
```

Speculation only stops paying below roughly 5x, and the measurement would have
to be off by 10x to get there. So the prefill-shaped verifying forward is not a
problem for this hardware -- it is the reason the approach works. Decode on
V100 is so thoroughly weight-bound that K extra token-slots are nearly free.

This is a ceiling, not a promise: it excludes the draft layer's own cost, which
is one extra layer per drafted token, and any verification overhead. But it
establishes that the remaining work is worth doing, which is what it was for.

## Verification: where it attaches in this codebase

Traced against the real code rather than restated from the review. The
mechanism is narrower than "a scheduler change", and upstream left the hook.

**Admission is driven by `seq_len`, and one field already carries "tokens I
submitted that are not in `seq_len` yet".**

```cpp
// request.h:415  -- ForwardTokenResource::InputLen
return s.seq_len + s.inflight_new_tokens - s.inflight_input_len - s.resume_len;

// engine.cc:455
context_length.push_back(c.seq_len + c.inflight_new_tokens /* plus draft tokens */);

// engine.cc:748
c.inflight_new_tokens = c.generating;   // exactly 1 for a decode row
```

That `/* plus draft tokens */` comment is upstream's, on the line that computes
context length. `inflight_new_tokens` is defined as "submitted generated tokens
not yet reflected into `seq_len`" (request.h:211), which is precisely what a
drafted token is. Setting it to `1 + accepted_draft_count` makes `InputLen`
return K+1, `ClampForwardEnd` admit K+1, and `s.input_len` become K+1 -- with no
change to the derivation I previously called a blocker.

**Acceptance is applied in `Engine::Update()`**, which today advances one token
per row:

```cpp
// engine.cc:718
c.token_ids[c.seq_len] = output_ids[j];
c.seq_len              = sequence_length[j];
```

`seq_len` is taken from the device-side `sequence_length[j]`, not incremented
locally. So the accepted count has to reach that buffer; the append loop then
follows automatically, since it already copies `c.seq_len - c.tokens.size()`
tokens rather than assuming one.

**The ordering that must hold**, and the reason it cannot be bolted on later:

1. Target forward scores K+1 positions.
2. Compare drafts against the argmax at each position; accept the longest
   matching prefix.
3. Set `sequence_length[j]` to the accepted end. Only now may `Update()` run.
4. Publication and checkpointing must not have happened for positions beyond
   the accepted end.

Step 4 is the one that escapes the request. `ClampForwardEnd` can return an
earlier `aligned` position for checkpoint publication (scheduler.cc:1008-1011),
and a checkpoint published at a speculative position that is later rejected
puts unverified KV into the prefix cache, where another request reuses it. The
MTP slot lives in that same registered prefix object, so this is not
hypothetical.

**A K+1 forward is a prefill, not a decode.** This changes the cost side of the
trade and is worth knowing before implementing, not after measuring a
disappointing speedup:

```cpp
// unified_attention_layer.cc:310 -- rows split by input_len
d.decode.n = find_if(rc, [](auto r) { return r->input_len > 1; }) - rc.begin();

// context_token_resource.h:50 -- temp memory
return (input_len > 1 || !s.is_active) ? ContextLen(s) : 0;
```

A speculative row has `input_len == K+1 > 1`, so it is counted in `prefill`,
dispatched through `invokeProcessKV_v2_` + `dispatchAttention` rather than
`dispatchDecoding`, and charged full `ContextLen` temp memory instead of zero.

So the comparison is not "K+1 decode steps for the price of one". It is one
prefill-shaped forward of K+1 tokens against K+1 decode forwards. On V100 that
is usually still favourable, because decode is memory-bound on weight traffic
and a K+1 prefill re-reads those weights once instead of K+1 times -- but it is
a different claim, and the 1.87 ceiling does not account for it. The realised
speedup must be measured end to end, not inferred from accept length.

It also means the pure-decode guard on the draft path (`d.all_decode`) will
exclude the verifying batch itself once verification lands: a batch containing
a K+1 row is not pure decode. Drafting for the next step has to happen on that
same forward, so that guard needs revisiting at the same time, and this is
exactly the kind of interaction that would otherwise surface as a Check failure
deep in attention.

### The write path, traced to the line

`sequence_length` is advanced in exactly one place, and it is not in the engine:

```cpp
// kernels/sampling_kernels.cu:64
if (tid == 0) {
    sequence_length[batch_id] += 1;
}
```

Everything downstream reads it. `stop_criteria` tests it against the length
limit, `logits_processor` uses it to index token history, `AppendTokenIds`
writes `token_ids[output_pos]` where `output_pos` was copied from it *before*
sampling, and `Engine::Update()` assigns `c.seq_len = sequence_length[j]`.

So verification does not need to touch `Engine::Update()` at all: writing the
accepted end into this buffer makes the append loop, the stop criteria and the
engine's own bookkeeping follow. `Update()` already copies
`c.seq_len - c.tokens.size()` tokens rather than assuming one, so it
generalises unchanged.

What must change, minimally:

1. `sequence_length[batch_id] += 1` becomes `+= accepted + 1`, or a separate
   post-sampling kernel adds the accepted count. The `+= 1` is inside the
   sampling kernel, which also runs for non-speculative rows, so this wants a
   per-row count defaulting to zero rather than an edit to the increment.
2. `output_pos` is snapshotted before sampling; with K+1 positions each
   accepted token needs its own slot, so `AppendTokenIds` must write a run.
3. `engine.cc:748` sets `inflight_new_tokens = c.generating`; it becomes
   `c.generating ? 1 + accepted : 0`.

That is three touch points, all local, none of them the scheduler rewrite I
originally described.

### The cross-request hazard has an existing guard

I traced (D) rather than trusting my own worry about it, and the codebase
already blocks the dangerous case:

```cpp
// scheduler.cc:963  -- PlanPublication
if (generation_cache_mode_ == CacheMode::kNone && end > s.prompt_len) {
    return;   // no checkpoint publication past the prompt
}
```

Speculative positions are by definition `> prompt_len`, since they occur during
generation. So with `cache_generation='none'` no speculative position can ever
be published, and the "rejected draft poisons another request's prefix" failure
is impossible by construction rather than by my remembering to prevent it.

That makes the safe first increment concrete: `cache_generation='none'` plus
`enable_prefix_caching=False`. The verify script currently sets neither, so it
runs on defaults (`auto`), and this must be set explicitly before the first
verifying forward -- not left to a default that could change.

Separately, the KV written for rejected positions is not read back: the next
step's `cu_k_len` is `PrefixSum(sequence_length)`, which stops at the accepted
end, so attention never reaches those bytes. They are stale but unreachable,
and the next forward overwrites them. That is only true while they are also
unpublished, which is what the guard above secures.

### Inference is correct, not merely self-consistent (commit bc2b09ec)

Every earlier check compared speculative text to baseline text and asserted
byte-identity. Two identical wrong answers pass that test. Measured against
known-correct answers instead, with chain-of-thought stripped so the model's
reasoning cannot satisfy the regex on the final answer's behalf:

```
drafting OFF: 15/15 checks passed over 15 generations
              latency early 0.82s vs late 0.69s
              VERIFY_INFERENCE_PASS

drafting ON : 15/15 checks passed over 15 generations
              latency early 1.17s vs late 0.97s
              VERIFY_INFERENCE_PASS
```

Five factual prompts (first five primes, 17x3, capital of Japan, powers of
two, "cat" reversed), three rounds in one process at budgets 128/256/64.
Every generation produced a correct answer, no degeneracy, and greedy
determinism held across rounds. Latency did not drift upward, so nothing leaks
across generations.

The checker was validated against negative controls before use: it correctly
rejects a wrong fifth prime, a wrong product, Kyoto for Tokyo, a wrong
sequence term, and an unreversed word.

### The non-zero exit was never a crash

The job reported `rc=1` for both text runs -- not 134 (SIGABRT), not 139
(SIGSEGV). Nothing crashed. "Aborted during teardown" was an explanation I
asserted all session without ever printing the status. rc=1 with no traceback,
after `main()` returns 0, is interpreter shutdown finalising the pipeline; the
generation itself completes, writes text, finishes with reason "stop" and
reaches `~Impl()`.

The job now asserts a positive `EMIT_TEXT_COMPLETE` marker and fails with exit
8 if it is absent, so a non-zero rc is tolerated only with proof the work
finished -- instead of being waved through on the assumption that it was
teardown, which would also have passed a run that died mid-generation beside a
stale text file.

### Settled by probe: the MTP slot holds zeros, not another sequence's bytes

```
unshifted,  32 samples: 18/32 = 56.2%, echo 3.1%, distinct 25
shifted -8, 32 samples: 18/32 = 56.2%, echo 3.1%, distinct 25
```

Moving the base eight positions down changed nothing, on a clean run with no
illegal access. With the earlier +1 result that is a nine-position range over
which the absolute base is irrelevant. Only zeros behave that way: attention
over zero-valued K/V contributes a constant however many entries are spanned.
Another sequence's leftover bytes would not survive being read at a different
offset.

Also measured, and it corrects two claims made earlier in this document:

```
step-1 acceptance 41/64 = 64.1%, distinct drafts 39
capacity cap by seq_len: 64->0, 65->4, 125->3, 126->2, 127->1, 128->0
```

**39 distinct tokens across 64 predictions** rules out an inert head emitting a
near-constant token (which would show 1-3). And the cap tracks `seq_len`
exactly as the modulo predicts -- the slack computation was always correct, and
a once-only warning latch made it look permanently stuck at zero. "The draft
attends to nothing" was wrong; 64.1% is a real number.

What remains true is narrower: the MTP slot is not seeded with the target's
history, so the draft attends to its own entries over a field of zeros rather
than over real context. That is a quality ceiling on deep steps, not a
correctness or safety defect, and it is why step 4 sits near zero.

### A fix that changed nothing, and why that is informative

Codex's highest-confidence P1 (0.98) was that step 0 writes at `L-1` rather than
`L`, because `k_offsets` is a `PrefixSum` taken before sampling increments
`sequence_length`. The ordering is real and the diagnosis is correct.

I fixed it, predicting in the commit message that deep steps would improve.
**They did not.** Steps 1-3 came back bit-identical (41/64, 15/63, 9/62); step 4
moved 1/61 -> 2/61, which is noise at that denominator.

Bit-identical under greedy decode means the change was a no-op, and the reason
is worth keeping:

```
before: step0 @L-1  step1 @L    step2 @L+1  step3 @L+2
after : step0 @L    step1 @L+1  step2 @L+2  step3 @L+3
```

Every position moved by exactly +1 -- uniformly. Step *k* reads `[0, cu_k_len)`
and writes at `cu_k_len - q_len`, so it still sees exactly the *k* entries that
steps `0..k-1` wrote. The *relative* geometry is unchanged; only the absolute
base moved. Shifting the base matters only if the entries below it carry
meaning -- and per the standing NOTE at language_model.cc:698, **the MTP KV slot
is never populated with target history at all**. Reading `[0,L)` of
uninitialised memory is indistinguishable from reading `[0,L-1)` of it.

So Codex's P1 is genuine but currently *masked* by its own other P1
(uninitialised MTP history, confidence 0.99). The position fix is a
prerequisite that cannot pay off until the history it indexes into exists. It
is kept, not reverted: it is correct, and it must be in place before seeding
history or the seeding would be written to shifted positions.

The cost of keeping it is measured, not assumed: `max_extend` goes from `K-1` to
`K`, so capacity-limited decode steps rise from 4.7% to 6.2%.

**The prediction failing is the useful part.** Had deep steps improved, I would
have credited the position fix and moved on with the far larger defect --
unpopulated draft history -- still in the tree, now hidden behind a number that
had gone up.

### My own guard becomes a silent 48% tax

`d.all_decode` (language_model.cc:454, `d.all_decode &= c.input_len == 1`) was
the right guard for a draft-only path: drafting needs a pure decode batch. Once
verification lands it turns into a self-cancelling oscillator:

```
step 1: no drafts pending -> input_len 1   -> all_decode true  -> draft
step 2: drafts pending    -> input_len K+1 -> all_decode false -> NO draft
step 3: no drafts pending -> input_len 1   -> all_decode true  -> draft
```

Speculation fires on alternate steps. At the measured numbers that is 1.74x ->
1.38x, **losing 48% of the gain**.

The reason to write this down now, before it happens: it produces no failing
test. Output stays byte-correct, acceptance stays 64.1%, every assertion passes.
Only throughput suffers, and only against a counterfactual that is never
measured. It is the same failure shape as the four defects in the last review
-- valid-looking, check-passing, silently wrong -- except this one is mine and
is already in the tree.

The fix is not to delete the guard. Drafting genuinely requires knowing which
row position holds the last accepted token, which is why it was restricted to
pure-decode batches in the first place. The correct form is "every row is
either decode or a *verifying* row", with the draft reading the hidden state at
the accepted end rather than at position 0 of the row.

**Smallest honest increment:** `bsz==1`, greedy, prefix caching off,
`cache_generation='none'`, K=1.
Under those constraints steps 1-3 are local and step 4 is satisfied by
construction, since publication is disabled. That is enough to produce a
measured accept length with byte-identical output -- the first result that
would mean speculation actually works rather than that it could.

## Headline: ~1.87 tokens per target forward (256-token run, commit 174e8165)

```
draft step 1: 41/64 = 64.1%  | given correct prefix: 41/64 = 64.1%
draft step 2: 15/63 = 23.8%  | given correct prefix: 11/40 = 27.5%
draft step 3:  9/62 = 14.5%  | given correct prefix:  3/10 = 30.0%
draft step 4:  1/61 =  1.6%  | given correct prefix:  0/2  =  0.0%
```

The position advance shows clearly at depth once the sample is large enough:
**step 3 raw went 3.3% -> 14.5%**, and step 2 went 16.1% -> 23.8%. These are
greedy runs on a fixed prompt, so the movement is deterministic, not variance.

Chaining the conditional rates gives the acceptance distribution:

```
P(>=1 accepted) = 0.641
P(>=2 accepted) = 0.176
P(>=3 accepted) = 0.053
P(>=4 accepted) = 0.000   (denominator 2 -- carries no information)

expected accepted drafts per step = 0.87
tokens per target forward         = 1.87
```

**This is a ceiling, not a speedup.** It is what speculation would buy if
verification were free. It is not free: the target must score K+1 positions per
forward, which costs more than a single-token decode, and none of that
machinery exists yet. The realised number will be lower and could be below 1.0
if verification is implemented badly.

An arithmetic check worth keeping, because it looked like an instrument bug at
first: each conditional denominator is one less than the previous step's hit
count (40 vs 41, 10 vs 11, 2 vs 3). That is end-of-run truncation, not an
off-by-one -- the final draft sets never receive all K ground-truth tokens. The
raw counts taper identically (64, 63, 62, 61), which is what confirms it.
Computing the chain from conditional rates rather than raw counts gives 1.87
either way, so the estimate does not depend on that subtlety.

Remaining caveats: one prompt, greedy, bsz=1, step-4 statistics are empty, and
the KV capacity guard still truncates the drafted range at block boundaries.

### What is actually limiting acceptance

Worth sizing before optimising the wrong thing. With `block=64` and `K=4`, the
capacity guard shortens the drafted range on 3 of every 64 decode steps --
about **4.7%**. Allocating one spare block per sequence would recover it.

The dominant term is elsewhere: conditional accuracy falls from 64.1% at step 1
to **27.5%** at step 2. That single drop is what caps the chain at 1.87, and no
amount of block-boundary work touches it.

Whether 27.5% is a defect or simply this MTP layer's quality is not yet
established. One candidate defect remains: the MTP slot accumulates the *draft
layer's* own K/V per decode step, never the target layer's, so the draft
attends to a history it wrote itself rather than to the target's. Published MTP
implementations condition on the target's hidden states. That is the next thing
worth testing, and it is a substantially larger change than advancing offsets.

Order of work, by expected value:

1. **Verification** -- without it there is no speedup at all, only a ceiling.
2. **Target-state conditioning** -- may raise step-2 conditional accuracy,
   which is the term that dominates the chain.
3. **Spare KV block** -- worth ~4.7% of steps; cheap, but do it last.

## Position advance: step 2 improves, and the metric is the wrong one

After `84d9631f` (commit 84d9631f, run 20260826_155803):

```
            before        after
step 1     56.2%   ->    56.2%     unchanged, as required
step 2     16.1%   ->    22.6%     +6.5 points
step 3      3.3%   ->     3.3%     unchanged
step 4      0.0%   ->     0.0%     unchanged
```

Step 1 held exactly, which was the stated guard: a change that moved it would
have been touching something it should not. Step 2 improved, and with greedy
decoding on a fixed prompt this is deterministic, not noise.

The capacity warning fired, and reconciling it was necessary rather than
dismissible. Prompt is 63 tokens with `block=64`, so slack per decode step runs
1, 0, 63, 62, ... -- the warning is emitted once, at the single step where
`seq_len == 64` leaves zero slack. Every later step permits the full advance.
So the guard fired truthfully and the advance is active for nearly all steps;
the two facts are consistent.

**Why steps 3 and 4 did not move, and why that is not a bug.** Acceptance at
step k is conditional: step k can only be correct if steps 1..k-1 were all
correct, because each conditions on its predecessors' output. So the raw
per-step rate has a hard ceiling of the previous step's rate, and the numbers
compound:

```
step 2 = 22.6% of 56.2%  ->  ~40% conditional accuracy
step 3 expected  30 * 0.562 * 0.40^2 = 2.7 hits   measured 1
step 4 expected  29 * 0.562 * 0.40^3 = 1.0 hits   measured 0
```

At this sample size steps 3 and 4 have **fewer than three expected hits
between them**. Measuring 1 and 0 is entirely consistent with a correctly
working predictor; those cells carry almost no information either way.

This means the raw per-step rate is the wrong instrument for the tail. The
right one is **conditional accuracy** -- of the cases where the prefix was
correct, how often is step k correct -- which does not shrink with depth and is
what actually predicts speculative speedup. The current numbers cannot
distinguish "step 4 is broken" from "step 4 was rarely reachable", and I should
stop reading them as though they can.

## Per-step acceptance confirms the diagnosis (commit c1e550d6)

```
draft step 1: 18/32 = 56.2%
draft step 2:  5/31 = 16.1%
draft step 3:  1/30 =  3.3%
draft step 4:  0/29 =  0.0%
```

This is the predicted shape. Step 1 is the only draft step whose attention sees
a correct history, and it is the only one that performs.

Some decay is expected even from a correct implementation, because errors
compound: step k conditions on k-1 of its own guesses. A healthy MTP layer
typically decays gently -- roughly 55/35/25/18. Falling to **zero** by step 4 is
not compounding error, it is the drafted positions being wrong.

Mechanism, now supported by measurement rather than assertion: `cu_k_len` is
fixed for the whole `Draft` call, so every step writes its K/V to the same
position and reads a history that stops at the target's last real token. Step 1
is correct by accident -- at that point the fixed offsets *are* the right ones.
Step 2 attends as though step 1 never happened, and so on.

The fix is therefore narrow and its success criterion is already defined: steps
2..4 must rise, while step 1 stays near 56%. A change that moves step 1 is
touching something it should not.

## Correction 3: the MTP slot does accumulate decode history

Tracing the seeding work turned up a third error in my own account.

`k_offsets` is a prefix sum over `sequence_length_`, recomputed every forward
(`language_model.cc:575`), and `sequence_length_` advances by one on every
decode step. The MTP attention consumes `params.cu_k_len = d.k_offsets`, the
same buffer the target uses. So on each decode step the draft writes its K/V at
the *current* sequence position, one step further along than the last.

The MTP slot therefore holds one genuine entry per decode step already: an
unbroken run of draft-step-0 states at exactly the positions the target's own
tokens occupy. It is not the target's history -- these are the draft layer's
projections, not the target layer's -- but it is coherent, correctly ordered,
and it grows.

That explains 56.2% far better than "the current token is in cache" alone, and
it revises the seeding plan again:

- **Wrong:** "nothing is ever written to the MTP slot."
- **Wrong:** "every draft step writes the same cache position." Across *decode
  steps* the position advances correctly.
- **Right, and now the only real defect:** within one `Draft` call, steps
  1..K-1 reuse the step-0 offsets, so the K drafted tokens all land on the same
  position and each sees a history missing its predecessors.

So the work is smaller and better targeted than "seed the KV": the per-decode
history is already correct, and what is missing is advancing the position
*within* a multi-step draft. That is also exactly why step-1 acceptance is
healthy while K>1 would not be -- a prediction the measurement now supports
rather than merely asserts.

## Correction 2: "attending to nothing" was too strong

The first acceptance measurement came back **56.2%**, having predicted near
zero. A number that far from the prediction indicts the analysis, the
instrument, or both. The analysis was wrong, and here is the mechanism.

`AttentionUniversal::operator()` stores the current token's K/V into the cache
*before* attending:

```cpp
if (ti >= readonly_len) {
    iterator.block_head_.with(iterator.block_ptrs_, local_ti, [&](auto k_cache, auto v_cache, ...) {
        Store(&k_cache[di], out_K[0][c]);
        if constexpr (HAS_V) { Store(&v_cache[di], out_V[0][c]); }
    });
}
```

The SM70 decode kernels (`kernel/decoding_sm70_*.cu`) instantiate exactly this
template, so the draft's own token *is* written to the MTP slot and *is*
visible to its own attention. The draft is therefore not attending to nothing:
it attends to itself, plus whatever the allocator left in the positions the
target's history should occupy.

So the earlier claim needs splitting in two:

- **False:** "the draft attends to uninitialised memory, so acceptance is
  meaningless." The current position is real.
- **Still true:** prior positions were never written by the MTP layer, because
  `UnifiedDecoder::Forward` runs only the target's layers. And `cu_k_len` never
  advances across draft steps, so steps 1..K-1 keep overwriting one slot.

That is enough to explain a non-trivial step-1 rate -- step 1 is the step that
depends least on history -- while steps beyond it stay unreliable. It also
reframes the seeding work: it should raise step-1 modestly and matter far more
for K>1, rather than moving the rate off zero.

**Settled by the controls** (commit e03a889d, run 20260826_153744):

```
step-1 draft acceptance: 18/32 = 56.2% (echo 3.1%, target-repeat 0.0%)
```

The echo hypothesis is dead. The predictor emitted its conditioning token on
1 of 32 positions, and the target repeated a token on **none** of them. So not
a single one of the 18 accepted drafts can be explained by repetition -- the
maximum repetition could contribute here is zero. The draft model is genuinely
predicting the target's next token roughly half the time.

Caveats that keep this honest:

- n=32, one prompt, greedy. This is a real signal, not a benchmark.
- Step 1 only. It says nothing about steps 2..K, which are precisely the steps
  the broken `cu_k_len` advance would damage.
- The predictor still has no target history in its slot, so this is a floor
  rather than the model's capability.

That last point is the useful one: **56.2% is what the draft achieves with only
its own token in cache.** Seeding the history should raise it, and if seeding
lands and the number does not move, the seeding did not work.

## Correction: the KV slot IS byte-isolated, but it is never filled

I traced the cache path to settle this properly, and both my earlier claim and
the review's phrasing were partly wrong.

What is true: `UnifiedAttentionLayer::Register` walks its weight list and gives
each weight its own `cache_block_offset` inside the block, then registers the
total as one `prefix_cache_offset_`. Attention reads KV through
`BlockIteratorParams{..., (int)weights.cache_block_offset, ...}`, i.e. the
offset carried by *that weight*. Because `unified_decoder.cc` pushed the MTP
attention onto that same weight list, the draft layer received genuinely
distinct bytes. So a discarded draft cannot scribble on a target layer's KV.
Item B, as originally posed, is answered: no corruption of the next decode.

What is false is that this makes the slot *usable*. Two things:

1. **Nothing ever writes it.** `UnifiedDecoder::Forward` loops over
   `weights.at(layer)` for `layer < layer_num_` -- the target's layers only.
   The MTP weight is in the registration list but not in the execution loop, so
   no prompt or decode history is ever written into its bytes. The first draft
   attention reads whatever the allocator last left there.

2. **Every draft step reuses one position.** `params.cu_k_len` comes from
   `d.k_offsets`, borrowed in `Setup` from the environment and fixed for the
   whole batch. `Draft` calls `DecodeStep` K times without touching it, so all
   K steps present the same history length and write the same cache slot.

So the draft attends to uninitialised memory at a position that never advances.
It cannot crash and cannot corrupt anything, which is precisely why it would
have survived a smoke test: drafts are discarded, output is unchanged, and any
accept length measured against it would be noise reported as a result.

Seeding and advancing this state is therefore not a follow-up detail. It is a
precondition for the first honest acceptance measurement.

## The MTP KV slot is inside the shared prefix object

This one came from external review and I had missed it entirely.

`unified_decoder.cc` obtains the draft layer's KV slot by pushing its attention
weight onto the same `attn_weights` vector the target layers register with.
That is what makes the registry allocate a slot at all -- but it also means the
draft layer's bytes live inside the *same registered prefix object* the prefix
cache publishes and other requests reuse.

So speculative KV is not automatically private. It is a region of a block that
prefix caching can hand to a different request. Two consequences:

1. Bytes written by a *rejected* draft are unproven. They must never be part of
   a published prefix. Publication intent has to stay empty for any speculative
   pass, not merely be corrected afterwards.
2. Even for an *accepted* draft, the MTP scratch bytes were produced by the
   draft layer rather than by a verified target forward. Before any future
   publication they must be rebuilt, or the MTP scratch must be moved to a
   cache category that is never reused.

This is the same hazard as checkpoint publication, one level lower: it escapes
the speculating request. The earlier note below reasoned about checkpoints and
assumed the KV slot was separate. It is not.

## Rollback: v0.16.0 already has the primitive

The fork's approach, a pair of shadow buffers with explicit Snapshot and
Restore, does not fit this tree, and it does not need to. v0.16.0 keeps the
GDN recurrent state in registry-allocated cache blocks rather than in flat
device buffers:

```cpp
// GatedDeltaNetLayer.cc
rec_base_ = registry.checkpoint().Register({{block_bytes_, 1, num_blocks_}});
registry.checkpoint().Register(conv_total_bytes_, 1);
```

So the state is already a `checkpoint` category object, and the scheduler
already knows how to copy one block into another:

```cpp
// scheduler.cc, restoring a checkpoint into the live frontier
s.restore_copies.push_back({best.ckpt, s.frontier.get()});
```

Those copies are executed by `model_executor.cc` via `RunCopies`. The
machinery exists because a resumed request must restore recurrent state that
a prefix-cache hit cannot reconstruct, which is the same problem a rejected
draft creates.

That makes item 4 a matter of reusing `restore_copies` with a
speculation-owned checkpoint block, rather than porting the fork's buffers.
The fork's design is still worth recording, below, because it names the two
traps: the conv and recurrent states have different element types, and the
shadow memory should not be allocated when speculation is off.

## How the fork rolls back GDN state

The fork does solve this, in `gated_delta_net_layer.cc`, and our v0.16.0
`GatedDeltaNetLayer` has none of it. A search for `snapshot` in our file
returns zero matches, so this is a port, not a reuse.

The fork keeps a shadow copy of both recurrent buffers and swaps them
wholesale:

```cpp
SnapshotState()   // conv_state -> conv_state_snapshot, recurrent -> snapshot
RestoreState()    // copy back, then mark the snapshot spent
DiscardSnapshot() // drafts accepted, keep the mutated state
```

Two details matter for the port. The two buffers have **different element
types**, `half` for the convolution state and `float` for the recurrent
state, so a single-dtype copy would silently corrupt one of them. And the
buffers are allocated **only when speculation is enabled**, which keeps a
non-speculative run from paying for memory it never reads.

Snapshot cost is one device-to-device copy of both buffers per draft run,
against 48 linear-attention layers. That is the price of rollback, and it is
charged whether or not the drafts are accepted, so it counts against the
speculation win.

## Work item 1, verified

```text
full-attention layers  : 16
=> expect MTP at index : 16
[MTP] registered a KV slot for the draft layer at attention index 16 of 17   x4 ranks
parsed: index=16 total=17
VERIFY_MTP_KV_SLOT_PASS
```

The draft layer sits at index 16, one past the target's last, on all four TP
ranks, and the model still generates coherent text. So the slot exists, it
does not overlap a target layer, and adding it did not disturb the target.

Four runs were needed, and the first three failed on the driver rather than
the code:

1. A checkpoint path that does not exist. The job named
   `Qwen3.5-27B-A3B-FP8`; the mounted model is `Qwen3.8-27B-FP8`.
2. Reading layer fields from the top of `config.json`. This checkpoint is
   `Qwen3_5ForConditionalGeneration`, so they live under `text_config`, and
   the top level returned `None` for all of them. That produced the message
   "this checkpoint declares no MTP layer", which is exactly what a genuinely
   MTP-free checkpoint would print. The driver was blaming the model for its
   own defect.
3. `%d` in the log line. `TM_LOG_INFO` forwards to `fmt::format`, so the
   specifier was printed literally. The registration had been firing on all
   four ranks the whole time; only the message was unreadable.

Worth keeping in mind: each of those failures looked like a finding about the
model, and none of them was.

## The layer split, settled

The checkpoint has 64 layers at `full_attention_interval: 4`. HuggingFace's
`Qwen3_5TextConfig.__post_init__` derives the pattern:

```python
"linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
```

That gives **16 full-attention layers and 48 linear**. An earlier note in
`SPECULATION.md` said 17 full, which sums to 65 for a 64-layer model and is
wrong. It is corrected there.

This is load-bearing rather than trivia: the KV slot verifier asserts the
draft layer lands at attention index 16, one past the target's last. Had the
17 been right, the assertion would have been off by one and would have failed
against correct code.

## Status

| item | state |
| --- | --- |
| 1. KV slot for the draft layer | **done and verified on hardware** |
| 2. `MTPPredictor` class | interface and fc projection committed; attention, FFN and argmax outstanding |
| 3. Draft loop | not started |
| 4. Verify and roll back | design known, GDN snapshot must be ported |
| 5. Accept-length measurement | not started |
