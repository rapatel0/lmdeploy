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
| 1. KV slot for the draft layer | implemented, compiles for SM70, verification running |
| 2. `MTPPredictor` class | interface and fc projection committed; attention, FFN and argmax outstanding |
| 3. Draft loop | not started |
| 4. Verify and roll back | design known, GDN snapshot must be ported |
| 5. Accept-length measurement | not started |
