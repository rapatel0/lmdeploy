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
