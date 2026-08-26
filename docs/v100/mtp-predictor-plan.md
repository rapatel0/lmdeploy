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
