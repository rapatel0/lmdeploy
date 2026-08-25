# Closing the speculative gap

The SGLang comparison in `north-star-gap.md` splits the deficit into two
multipliers. This file covers the larger and better understood one:
speculative decoding, worth 2.20x in SGLang's own measurement.

The useful finding is that LMDeploy already has most of the machinery. The
campaign plan assumed otherwise.

## LMDeploy already implements speculative decoding

The Phase 2 MTP audit looked at the donor's `MTPPredictor` and concluded a
future campaign would have to design acceptance and rejection from scratch,
because the donor supplies draft generation only.

That conclusion was about the donor. It is wrong about the product.

`lmdeploy/pytorch/spec_decode/proposers/` contains six registered proposers:

```text
deepseek_mtp.py   eagle.py   eagle3.py
hy3_mtp.py        qwen3_5_mtp.py        base.py
```

`SpeculativeConfig` is a public API surface with `method`, `model` and
`num_speculative_tokens`. The CLI already parses `--speculative-algorithm`.
`BaseSpecProposer` is a 174-line contract with `build_model`, `_forward`,
`update_inputs_decoding`, `embed_input_ids`, `get_logits` and
`get_target_hidden_size`.

Critically, `config_builder.py` gives `qwen3_5_mtp` the target's own
`DistConfig` rather than a fresh one, so that method already runs the draft
model under the same tensor-parallel layout as the target:

```python
if speculative_config.method in ('qwen3_5_mtp', 'hy3_mtp'):
    draft_dist_config = dist_config
```

Other methods fall back to `DistConfig()`, and the comment there is explicit
that TP above one is not yet supported for them.

## The catch: it is the PyTorch backend, not TurboMind

Every symbol above lives under `lmdeploy/pytorch/`. The campaign runs on
TurboMind, which has no speculative path at all. `lmdeploy/turbomind/` mentions
drafter layers only in `checkpoint.py`, for skipping them during conversion.

So the choice is not "implement speculative decoding". It is:

1. Run the campaign target on the PyTorch backend, where speculative decoding
   already exists, and find out what it costs on SM70.
2. Add a speculative path to TurboMind, which is the larger piece of work.

Option 1 is a measurement, not a port. It should happen first, because it
prices option 2.

## The draft model is already on the cluster

`/srv/models/Qwen3.8-27B-DFlash2`, 3.6 GB against the target's 28.8 GB, about
12 percent overhead.

Its `config.json` matters for the design:

```json
"architectures": ["DFlash2DraftModel"],
"dflash_config": {
  "block_size": 8,
  "selector_top_k": 16,
  "target_layer_ids": [5, 19, 33, 47, 61]
}
```

`target_layer_ids` means this draft model consumes hidden states from five
specific layers of the target. It is not a standalone small model that can be
run beside the target; it is coupled into the target's forward pass.

That is why `BaseSpecProposer.get_target_hidden_size` exists, and it is the
same shape of coupling the existing MTP proposers already handle. It is also
why a naive "run a small model as a drafter" design would not work here.

`block_size` 8 is consistent with SGLang's measured accept length of about
4.2: the draft proposes eight tokens per step and roughly half survive
verification.

## SGLang is the reference implementation, not the source

`sglang-V100/python/sglang/srt/speculative/` contains 4380 lines across the
dflash worker, info and utils modules. Read it for the acceptance rule, the
draft-token layout, and how the verify step is batched.

Do not port it. It is written against SGLang's scheduler, its
`ForwardBatch`, and its attention backend. The equivalent LMDeploy structures
already exist and differ.

## Proposed order

1. Measure LMDeploy on the PyTorch backend at batch 1 with
   `--speculative-algorithm` against the DFlash2 draft, on island 2. Compare
   with the TurboMind batch-1 number and with SGLang's 117.58 tok/s.
2. If PyTorch-backend speculation works on SM70, the remaining question is
   whether TurboMind's throughput advantage outweighs the missing 2.2x. That
   is a measurement, and it decides where the rest of the campaign spends
   effort.
3. Only then consider a TurboMind speculative path.

This order costs one job to answer a question the plan currently answers by
assumption.

## What this changes in the plan

The specification defers MTP to a separate campaign and forbids implementing it
here. That instruction was written on the belief that MTP meant porting the
donor's incomplete `MTPPredictor`.

The product already ships a working, TP-aware speculative framework with six
proposers. The gap between that and the north star is a backend question, not
an implementation-from-scratch question.

Raise this with the operator before proceeding, because it changes the
campaign's priority order rather than just its task list.
