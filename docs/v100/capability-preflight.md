# Phase 0A capability preflight

This report records the static capability checks that the master specification
requires before donor work.

Every check in this report ran against the locked product base. No check ran on
a GPU. The report marks each GPU-dependent item as unverified.

| Field | Value |
| --- | --- |
| Product base | `v0.16.0` = `1208bf006bbac69f1f012ceafeeeb70f623b632c` |
| Branch | `experiment/lmdeploy-v100-next` |
| Method | Static source inspection and checkpoint metadata reads |

## Checkpoint recognition

The target checkpoint exists at `/srv/models/Qwen3.8-27B-FP8`.

| Property | Value |
| --- | --- |
| `model_type` | `qwen3_5` |
| Layers | 64 |
| Layer types | 48 `linear_attention`, 16 `full_attention` |
| Attention heads | 24 |
| KV heads | 4 |
| `head_dim` | 256 |
| `hidden_size` | 5120 |
| Max position | 262144 |
| Quantization | FP8 `e4m3`, `weight_block_size` `[128, 128]` |
| Multimodal | Yes, the config carries a vision section |

The specification states that Qwen3.8 full attention uses 16 layers, 4 KV
heads, and head_dim 256. The checkpoint metadata matches that statement.

The query group size is 24 / 4 = 6.

Read these values from `text_config`, not from the top level. The top level
holds only `text_config`, `vision_config`, and `quantization_config`.

## Tensor and scale inventory

| Group | Count |
| --- | --- |
| Total tensors | 1606 |
| Language model | 1250 |
| Vision tower | 333 |
| MTP | 22 |
| Scale tensors | 407 |

The scale tensors use the `weight_scale_inv` suffix. `weight_format.py` maps
that exact suffix in `FP8Format.suffix_map`, and `accepts()` fails closed when
the suffix is absent.

Tensor names confirm the layer split independently of the config: 16 layers
carry `self_attn` tensors and 48 layers carry `linear_attn` tensors.

## Attention shapes

Read from the safetensors header for layer 3.

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `q_proj.weight` | `[12288, 5120]` | `F8_E4M3` |
| `k_proj.weight` | `[1024, 5120]` | `F8_E4M3` |
| `v_proj.weight` | `[1024, 5120]` | `F8_E4M3` |

The K and V projections give 1024 / 256 = 4 KV heads, which matches the
config.

The Q projection gives 12288 / 256 = 48, which is twice the configured 24.
That is expected. The model uses an output gate. `qwen3_5.py` sets
`self._attn_cfg.output_gate = True` and calls `split_output_gate`, and the
PyTorch config doubles `num_attention_heads` when `attn_output_gate` is set.

The query group size is 6 either way, and both readings select `kH=3`.

## KV budget check

For 16 full-attention layers, 4 KV heads, and head_dim 256:

```text
FP16 KV        2 * 16 * 4 * 256 * 2 = 65536 bytes = 64 KiB per token
```

This confirms the figure that the specification states.

Block-64 INT8 costs 34816 bytes for each token, a 47 percent saving against
FP16 before layout alignment.

## Architecture mapping

| Capability | Result | Evidence |
| --- | --- | --- |
| Qwen3.5 architecture mapping | PASS | `lmdeploy/turbomind/models/qwen3_5.py` reads `cfg.layer_types[i] == 'linear_attention'` |
| Gated DeltaNet | PASS | `lmdeploy/pytorch/nn/gated_delta.py`, `kernels/cuda/gated_delta_rule.py`, `kernels/cuda/gated_delta_preprocess.py` |
| FP8 weight loading on SM70 | PASS | See the next section |
| Full-attention ownership | UNVERIFIED | Needs a runtime check |
| CUDA graph support | UNVERIFIED | Needs a runtime check |
| OpenAI-compatible serving | UNVERIFIED | Needs a runtime check |
| TP1 startup, small model | UNVERIFIED | Needs GPU |
| TP4 startup, Qwen3.8 | UNVERIFIED | Needs GPU |

## FP8 weight-only on SM70

The base supports FP8 weights on SM70 with FP16 compute.

`kernels/gemm/arch/config_sm70_s884.h` defines:

```text
Config_E4M3 = Sm70_s884<Operand_A<half>,            A
                        Transform_Default,          transform A
                        VoidOperand,                U
                        Operand_B_Pack<fp8_e4m3_t>, B
                        Transform_HMMA_SIMT_B,      transform B
                        Operand_V_Pack<uint16_t>,   V
                        kRowMajor, half, ...>
```

`kernels/gemm/kernel/sm70_884_8.cu` instantiates `Config_E4M3` and registers
five tile shapes. `kernels/gemm/CMakeLists.txt` builds that file.

The path is registered and compiled. It is not dead code and not PTX-only.

This result removes foundation-port work from Phase 3.

It does not prove Qwen3.8 loader coverage, scale semantics, dense and MoE
route coverage, or FP8 KV attention support. Phase 3 still needs
qualification.

The SM70 GEMM FP8 result says nothing about attention KV data types. The two
subsystems are separate.

## SM70 head_dim 256 decode registrations

The base dispatches decode through a runtime registry, not a compile-time
config. `decoding.cu` builds an `AttnDesc` and calls `Registry::instance().Find`.

`kernels/attention/kernel/decoding_sm70_256.cu` registers nine kernels through
`MMA_SIMT`:

| Activation | KV type | Query-head tiles |
| --- | --- | --- |
| `half` | `half` | `kH` 1, 2, 3 |
| `half` | `uint8_t` | `kH` 1, 2, 3 |
| `half` | `uint4_t` | `kH` 1, 2, 3 |

Qwen3.8 has query group size 6, so the runtime selects `kH=3`. The kernel for
Qwen3.8 INT8 KV decode is `KT<half, uint8_t, 3>`, which the base registers
today.

SM70 head_dim 256 quantized KV decode is an existing route.

Registration proves that the runtime can select the kernel. It does not prove
correct output or linked SM70 machine code. The Phase 2 gate stays mandatory.

## Correction to an earlier finding

An earlier reading treated the donor `MMA_884_DEC` type constraint as a
campaign blocker. That reading was wrong.

The donor constraint is real: `impl_884_decode.h` declares `using Tkv = T_`,
so donor `MMA_884_DEC` cannot consume a KV type distinct from the activation
type.

That constraint does not apply to the product base. The base contains no
`MMA_884_DEC` symbol, no `decoding_config.h`, and no `codegen/` directory.
Those files are donor-only. The base superseded that design with the registry.

The donor report of incorrect TP8 output remains a risk to test on the base
SIMT path. It is no longer evidence that the only viable route is broken.

## Consequence for Phase 5

The base reader loads one `(scale, zero)` pair for each token and KV head.
`impl_simt.h` applies that single pair to every dimension fragment.

At head_dim 256, block-64 needs four parameter groups for each token and head.

A converter-only change would reuse one group's scale for three other groups,
which corrupts results silently.

Phase 5C must change the parameter fetch, the storage layout, and the writer
statistics, not only the converter.

## Hardware preflight

Recorded on `gpu-01`.

| Fact | Value |
| --- | --- |
| GPUs | 8 x Tesla V100-SXM2-32GB |
| Busy island | GPUs 0-3, about 31.4 GB used for each, four compute processes |
| Active workload | `sglang-dflash2-fp16-tp4-final` in namespace `llm` |
| Free island | GPUs 4-7, 0 MiB used, no compute processes |
| Free island topology | Full NVLink mesh among GPUs 4-7, `SYS` to GPUs 0-3 |
| Free island CPU affinity | 20-39, 60-79 |
| Free island NUMA node | 1 |

GPU UUIDs:

| Index | UUID |
| --- | --- |
| 0 | `GPU-3ceb3a71-cd56-6d10-075e-0300bd506c22` |
| 1 | `GPU-aa23eb12-62a1-161f-b566-3a8b5d0c6278` |
| 2 | `GPU-f364b813-c606-1786-40be-f6645f3c33eb` |
| 3 | `GPU-07e14590-993d-404b-4a47-f65d0c4b23e0` |
| 4 | `GPU-c1aa8bd9-f642-328d-96f8-79d7c38ce61e` |
| 5 | `GPU-c275f176-56b9-eb8d-14d7-a3fd15a0ec03` |
| 6 | `GPU-d5302639-be81-61f0-5e1a-b8f421bb100b` |
| 7 | `GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b` |
| | |

The free island matches the allocation that the specification describes.

Do not rely on these host indexes after a workload change. Re-verify before
each GPU run.

## Model selection

The operator selected `Qwen3.8-27B-FP8` as the campaign target.

### Why FP8 fits

| Checkpoint | Size on disk | Quantization | Note |
| --- | --- | --- | --- |
| `Qwen3.8-27B-FP8` | 29 GB | `fp8` `e4m3` | Selected baseline |
| `Qwen3.8-27B-BF16` | 28 GB | none | Downgrades to FP16 on SM70 |
| `Qwen3.8-27B-EXL3-6.00bpw` | 22 GB | `exl3` 6-bit | Not supported by TurboMind |
| `Qwen3.8-27B-EXL3-K5K6-context` | 20 GB | `exl3` 4-bit | Not supported by TurboMind |
| `Qwen3.8-27B-DFlash2` | 3.6 GB | none | Speculator, not a full model |

Only the FP8 and BF16 checkpoints are usable today. TurboMind has no `exl3`
loader, so both EXL3 variants are excluded regardless of their smaller size.

### Weight-format options on SM70

This table answers which other variants are worth building later.

| Weight path | SM70 GEMM config | Loader | Available for Qwen3.8 |
| --- | --- | --- | --- |
| FP16 dense | `Config_F16` | `trivial` | Yes, from the BF16 checkpoint |
| FP8 `e4m3` | `Config_E4M3` | `fp8` | Yes, selected |
| INT4 AWQ | `Config_U4_d`, `Config_U4_g` | `awq` | Not on disk, buildable |
| MXFP4 | `Config_MXF4` | `mxfp4` | Not on disk, buildable |
| INT8 weight-only | None | None | Not possible on SM70 today |
| EXL3 | None | None | Not supported |

The SM70 GEMM registers three quantized weight operands: `uint4_t`,
`fp4_e2m1_t`, and `fp8_e4m3_t`. There is no INT8 weight operand.

### INT8 weights and INT8 KV are different things

Do not confuse these two uses of INT8.

INT8 **weights** have no SM70 GEMM kernel. That path does not exist.

INT8 **KV cache** does exist. `decoding_sm70_256.cu` registers
`KT<half, uint8_t, kH>` for head_dim 256.

The campaign outcome is block-scaled INT8 **KV**, which sits on the attention
path, not the GEMM path. The absence of an INT8 weight kernel does not block
it.

A smaller weight format such as AWQ INT4 would free HBM for KV. That is a
separate lever from the KV format, and the specification keeps checkpoint
weight bytes unchanged during the campaign.

The checkpoint holds 30890012444 bytes, which is 28.8 GiB.

At TP4 the weight share is 7.2 GiB for each GPU. A V100-SXM2-32GB reports
32768 MiB, so about 24.8 GiB remains for KV, activations, and graphs.

TP1 does not fit. 28.8 GiB of weights leaves no room for KV on a 32 GiB card.
The specification already forbids Qwen3.8 at TP1.

### Tensor-parallel alignment

FP8 uses 128 by 128 blocks, so every sharded dimension must stay a multiple
of 128 after the split.

All ten distinct language-model weight shapes satisfy that rule at TP1, TP2,
TP4, and TP8. No projection needs a padded or separated commit.

### Precision on SM70

SM70 has no BF16 support. `is_bf16_supported()` requires compute capability 8
or higher, and `_resolve_dtype` falls back to `float16` with a warning.

The BF16 checkpoint therefore gives no precision benefit on V100. It converts
to FP16 and costs the same KV bytes.

`kernel/decoding_sm70_256.cu` registers only `half` activations, which agrees
with that fallback.

FP8 weights with FP16 compute is the correct trade for this hardware.

## Weight format on load

`FP8Format.pack()` keeps FP8 native and reports `TYPE_FP8_E4M3`.

`FP8Format.dequant()` expands FP8 to the target dtype, but `_dequant_linear`
runs only through `dequant_mixed`, which acts only when a fusion group holds
more than one weight format.

The SM90 tests in `builders/ffn.py` select fused-SiLU packing. They do not
gate FP8 support.

The expected SM70 behavior is native FP8 weights with FP16 compute, which
matches `Config_E4M3`. A GPU run must confirm the loaded weight bytes.

## Remaining work before GPU time

These items are now verified statically:

- Checkpoint tensor names and scale names.
- FP8 block shape, 128 by 128, matched by `FP8Format(block_in=128, block_out=128)`.
- Expected tensor counts.
- Layer-count mapping and full-attention layer identity.
- SM70 registry and build linkage.

These items still need verification:

- Ignored-module decisions for the vision tower and MTP heads.
- Gated DeltaNet state allocation sizes at run time.
- CUDA graph source paths.
- OpenAI serving paths.
- Loaded tensor counts observed at run time.

TP1 and TP4 startup need GPU access and operator approval.

## Qwen3.8-27B-FP8 at TP4 fails in the linear-attention out_proj

The model loads. Weights convert, all four ranks allocate, and the engine
reaches the first forward pass. It then aborts:

```
[TM][FATAL][gemm.cu:341] No feasible kernel found for the problem:
  sm70_f16_e4m3b128_f16_ttt_fff_8x5120x1536_1
  [1] LlamaLinear::Impl::Forward()      @ LlamaLinear.cu:148
  [2] linear_.Forward                   @ GatedDeltaNetLayer.cc:514
  [3] GatedDeltaNetLayer::Forward()     @ GatedDeltaNetLayer.cc:355
  [4] layer_0                           @ unified_decoder.cc:243
```

This is a real gap in the current source, not a build defect. The SM70 wheel
is sound: the same wheel runs Qwen3.5-4B at TP1 and TP4 and passes both gates.

### Causal chain

**The checkpoint quantizes `out_proj` but not the in-projections.**
`modules_to_not_convert` lists `linear_attn.in_proj_a`, `in_proj_b`,
`in_proj_ba`, `A_log`, `conv1d`, `dt_bias`, and `norm` for every linear
attention layer. It does not list `linear_attn.out_proj`, so that weight is
FP8 e4m3 with `weight_block_size` `[128, 128]`.

**The converter dequantizes only the in-projections.** In
`lmdeploy/turbomind/builders/deltanet.py`, `add_input_projections` runs
`dequant_mixed(...)` over the fused input projections, then adds the output
projection unchanged:

```python
fused = fuse_gdn(q, k, v, z, b, a, tp=self.tp.size)
self._add_linear('in_proj_all', fused, SplitSide.OUTPUT)
if out_proj is not None:
    self._add_linear('out_proj', out_proj, SplitSide.INPUT)
```

`dequant_mixed` only acts when formats differ among the linears passed to it,
and `out_proj` is not among them, so it stays FP8.

**The resulting GEMM has no SM70 kernel.** `linear_num_value_heads` is 48 and
`linear_value_head_dim` is 128, so `value_dim` is 6144. `out_proj` splits on
the input side, so at TP4 `K` is 1536. With `hidden_size` 5120 the problem is
`M=8, N=5120, K=1536`.

`Config_E4M3` for SM70 is declared in
`src/turbomind/kernels/gemm/arch/config_sm70_s884.h` and instantiated in
`src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` with five tiles:

```
<128, 128, 16>  <64, 128, 32>  <32, 128, 32>  <16, 128, 32>  <8, 128, 64>
```

Every tile uses `Operand_B_Pack<fp8_e4m3_t>`, meaning B must be pre-packed,
and `Operand_V_Pack<uint16_t>` for the scales. `Kernel::is_feasible` in
`src/turbomind/kernels/gemm/kernel.cu` compares `pack_b` and `pack_v` exactly,
so an unpacked FP8 operand cannot match any registered kernel regardless of
tile size. Alignment is not the obstacle: 5120 and 1536 are both multiples of
128.

### Why the small model passed

Qwen3.5-4B is unquantized, so its GDN `out_proj` is FP16 and dispatches to
`Config_F16`, which is registered for SM70 in `sm70_884_16.cu`. The TP1 and
TP4 gates therefore exercised the serving path and the NCCL collectives, but
never the FP8 weight path. That is exactly the limitation recorded when the
small model was selected.

### Options

1. Dequantize `out_proj` to FP16 in the converter, matching the treatment the
   in-projections already receive. Costs memory on the 48 linear-attention
   layers and changes no kernel.
2. Pack FP8 `out_proj` into the layout `Operand_B_Pack` expects, so the
   existing SM70 tiles apply.
3. Add an SM70 `Config_E4M3` variant that accepts an unpacked operand.

Option 1 is the smallest change and the natural next step, since the converter
already dequantizes sibling weights in the same function.

## Block-scaled FP8 is an SM90-only capability

Qwen3.8-27B-FP8 cannot run on SM70 without a new kernel. This is a hardware
capability gap in the GEMM library, not a configuration or packing problem.

### The decisive comparison

`MakeQuantDesc` in `src/turbomind/models/linear_weight.cc` classifies a weight
by its blocking:

```cpp
if (fmt.dtype == kFloat8_e4m3) {
    // Weight format has bidirectional blocking {128, 128} -> B-type.
    if (fmt.block_sizes.size() > 1 && fmt.block_sizes[1] > 1) {
        return {gemm::QuantType::kB, gs};
    }
    return {gemm::QuantType::kK, gs};
}
```

The checkpoint declares `weight_block_size` `[128, 128]`, so every FP8 weight
in this model resolves to `QuantType::kB`, meaning a two-dimensional block
scale.

Every kernel built from the generic `KernelImpl` declares the other type:

```cpp
desc_.quant_b = QuantDesc{QuantType::kDefault, OpV::kGroupSize};
// QuantType::kDefault == kK
```

`Kernel::is_feasible` compares the two exactly:

```cpp
if (desc.quant_b.type != desc_.quant_b.type) return false;
```

`kB` is 3 and `kK` is 1, so no SM70 kernel can match, regardless of tile size,
operand order, or packing.

`QuantType::kB` is declared in exactly one place in the entire kernel tree:

```
src/turbomind/kernels/gemm/kernel_impl_sm90.h:125:
    desc_.quant_b = QuantDesc{QuantType::kB, 128};
```

There is no pre-SM90 mainloop that consumes a two-dimensional block scale.

### What the two failed attempts established

**Attempt 1, dequantize the GDN output projection.** It worked, and the
failure moved from `GatedDeltaNetLayer.cc:514` to `LlamaFfnLayer.cc:58`.
`modules_to_not_convert` excludes only norms and gates, so every GEMM weight
in the model is FP8. Dequantizing them one at a time is equivalent to running
the model in FP16, at 57.6 GiB instead of 28.8 GiB, which discards the reason
for choosing the FP8 checkpoint. Reverted.

**Attempt 2, enable pre-SM90 weight packing.** The empty
`else if (weight_format.dtype == kFloat8_e4m3)` branch in
`LinearWeight::prepare` really did skip the converter, and removing it changed
the dispatch descriptor from `ttt` to `tnt`, which proves the SM70 converter
ran and packed the weight. The dispatch still failed, because packing was
never the binding constraint. Reverted.

Both attempts were necessary to locate the real constraint, and both are
recorded here so the next campaign does not repeat them.

### Consequences for the campaign

The FP8 checkpoint is unusable on V100 as-is. Three paths remain.

1. Use an INT4 AWQ or MXFP4 build of Qwen3.8-27B. Both are K-grouped, both
   have working SM70 kernels in `sm70_884_4.cu`, and the INT4 SM70 path is
   already proven in production on this cluster.
2. Serve the unquantized checkpoint as FP16 at TP4. Weights are 14.4 GiB per
   GPU, which fits a 32 GiB card, and every kernel needed already exists.
3. Write an SM70 mainloop that consumes a two-dimensional block scale. That
   is a genuine kernel project, not a configuration change.

Option 1 preserves the memory advantage and is the natural next step. Option 2
is the fastest route to a running baseline.
