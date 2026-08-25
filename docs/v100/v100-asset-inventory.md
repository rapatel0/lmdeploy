# V100 / SM70 Asset Inventory

Date: 2026-08-25

This file lists the reusable assets found in the five sibling repositories next
to `lmdeploy/`. Every entry below was read directly in the source tree. Claims
about behavior come from code or from committed benchmark records, not from
README summaries alone.

The immediate goal is Qwen3.8-27B-FP8 on four V100-32GB GPUs at TP4.

## Summary of the most important finding

Two repositories already serve block-scaled FP8 checkpoints on V100. Neither
one wrote a new block-scale mainloop. Both expand the scale at load time and
then use the ordinary K-grouped FP8 kernel.

`1Cat-vLLM` calls into a vendored copy of this repository's GEMM library, and
its `Config_E4M3` definition is byte-identical to ours. The kernels needed to
run this model are already compiled into our wheel.

## Repository map

| Repository | Size | Role for this project |
| --- | ---: | --- |
| `1Cat-vLLM` | 376M | Highest value. SM70 op surface plus a vendored TurboMind. Serves Qwen3.8-27B-FP8 TP4 on V100. |
| `sglang-V100` | 215M | Accepted benchmark numbers for the exact target model and hardware. |
| `lmdeploy-v100` | 13M | Earlier LMDeploy V100 port. Closest lineage to our tree. |
| `marlin_v100` | 6.9M | SM70 Marlin dense and MoE workspace. Independent confirmation of the scale technique. |
| `Tilelang-FA-V100` | 352K | TileLang FlashAttention for V100, including paged and backward kernels. |

## 1. The FP8 block-scale technique

This is the item that unblocks the current campaign.

A `{128, 128}` block scale means one scalar covers a 128-by-128 tile. A
K-grouped scale with group size 128 means one scalar covers a 128-by-1 column
slice. Expanding the block scale along N by 128 converts the first into the
second. Every weight element still receives the scale of its own block, so the
transform is exact.

`1Cat-vLLM`, inside the CUDA op `fp8_sm70_prepare`:

- File: `1Cat-vLLM/csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`, near line 2872.

```cpp
auto group_scales = scales.transpose(0, 1)
                        .contiguous()
                        .to(torch::kFloat16)
                        .repeat_interleave(group_size, 1)
                        .slice(1, 0, n)
                        .contiguous();
```

`marlin_v100`, in Python at load time:

- File: `marlin_v100/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`, near line 175.

```python
scales = scales.repeat_interleave(block_n, 1)
scales = scales[:, :part_size_n]
```

Both then declare the weight K-grouped, which routes to `QuantType::kK`.

### The converter call that proves reuse

`1Cat-vLLM` requests the converter from our own library:

```cpp
GetConverters(kHalf, kFloat8_e4m3, kHalf, /*grouped=*/true, 70);
```

That returns the SM70 line in our `convert_v3.cu`:

```text
W(sm70, kRow, s884h | B | _1), S(sm70, kCol, s884h | V | _1)
```

which packs for the `Config_E4M3` tiles in our `sm70_884_8.cu`.

## 2. GEMM kernels for SM70

Our tree, `1Cat-vLLM`, and `lmdeploy-v100` all carry the same three kernel
files. The configuration headers agree on operand types.

| File | Configs registered | Quant type | Present in our tree |
| --- | --- | --- | --- |
| `sm70_884_16.cu` | `Config_F16` | none | yes |
| `sm70_884_4.cu` | `Config_U4_d`, `Config_U4_g`, `Config_MXF4` | `kK` | yes |
| `sm70_884_8.cu` | `Config_E4M3` | `kK` | yes |

`Config_E4M3` is identical across trees:

```cpp
Operand_A<half>            // A
Transform_Default          // transform A
VoidOperand                // U
Operand_B_Pack<fp8_e4m3_t> // B
Transform_HMMA_SIMT_B      // transform B
Operand_V_Pack<uint16_t>   // V
kRowMajor                  // order_C
half                       // Tc
```

### Tile coverage differs, and theirs is wider

Our tree and `lmdeploy-v100` register 5 E4M3 tiles. `1Cat-vLLM` registers 13.
The extra tiles are mostly `CTA_N = 256` and additional small-M shapes.

Tiles present in `1Cat-vLLM` and missing from our tree:

```text
C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>
C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128,  48, 128>
C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>
C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>
C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>
C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>
C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>
C::Type<  8, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>
```

Copying these is low risk, because the `Config_E4M3` alias they instantiate is
already identical. Note the registration syntax differs: our tree uses a
`Registrar` with `c.add<...>()`, and theirs uses `Registry::sm70_884_8()` with
`Add<...>()`. Translate the syntax; do not copy the file.

Our tree also lacks `Config_NVF4`, which `1Cat-vLLM` defines alongside
`Config_MXF4`. That matters only if NVFP4 checkpoints become a target.

## 3. SM70 operator surface in 1Cat-vLLM

Registered in `1Cat-vLLM/csrc/torch_bindings.cpp`. These are ready-made and
already validated against the target model.

### Weight preparation

| Op | Use |
| --- | --- |
| `fp8_sm70_prepare` | Block FP8 to TurboMind K-grouped layout. The key op. |
| `awq_sm70_prepare` | AWQ INT4 to TurboMind layout. |
| `mxfp4_sm70_prepare` | MXFP4 to TurboMind layout. |
| `nvfp4_sm70_prepare` | NVFP4 to TurboMind layout. |
| `uint4_sm70_prepare` | Compressed-tensors INT4. |
| `sm70_f16_prepare` | Plain FP16 pack. |

`fp8_sm70_prepare` validates its inputs and requires `group_size == 128`, FP32
input scales, and 2-D scales. It returns the packed weight, the expanded
scale, and a metadata tensor holding `k_ld` and `q_ld`.

### Dense GEMM

| Op | Use |
| --- | --- |
| `fp8_gemm_sm70_out`, `_out_auto`, `_out_meta` | FP8 dense GEMM entry points. |
| `fp8_gemm_sm70_prefill_dispatch_out` | Large-M prefill route. See section 6. |
| `awq_gemm_sm70`, `_out`, `_out_tile_reduce` | INT4 dense GEMM. |
| `mxfp4_gemm_sm70_out`, `nvfp4_gemm_sm70_out` | FP4-family dense GEMM. |
| `sm70_f16_gemm`, `_out` | FP16 dense GEMM. |
| `sm70_f16_gate_mul_out` | Fused gate multiply. |

### MoE

A full FP8, INT4, and MXFP4 MoE surface exists, including per-expert dispatch,
dense-stage variants, and single-token fast paths. Relevant only if a MoE
checkpoint becomes a target. Representative names:
`fp8_moe_gemm_sm70_out`, `fp8_moe_gemm_sm70_per_expert_dispatch_out`,
`fp8_moe_single_token_dense_w13_sm70_out`.

### Fallback and utility

| Op | Use |
| --- | --- |
| `fp8_sm70_dequantize_out` | Dequantize FP8 to FP16 for unsupported shapes. |
| `awq_sm70_dequantize_out` | Same for INT4. |
| `sm70_gemm_export_cache`, `sm70_gemm_import_cache` | Persist GEMM autotune results across runs. |
| `sm70_tp2_all_reduce_gemma_rms_norm`, `sm70_tp4_all_reduce_gemma_rms_norm` | Fused all-reduce plus RMSNorm at TP2 and TP4. |
| `sm70_f16_lm_head_top1_out`, `_top1_tc_out`, `_top20_tc_out` | Fused LM head with top-k. |
| `sm70_sample_packed_top20_out`, `sm70_merge_tail_top20_pack_out` | Sampling helpers. |
| `sm70_marlin_available` | Runtime capability probe. |

## 4. Environment switches worth copying

From `1Cat-vLLM/vllm/envs.py`. These show which routes are considered safe
enough to enable by default, and give a rollback for each.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_SM70_FP8_TURBOMIND` | 1 | Block-FP8 dense path through TurboMind W8A16. |
| `VLLM_SM70_FP8_DEQUANT_FALLBACK` | 1 | Dequantize at load for shapes the dense kernel does not handle. |
| `VLLM_SM70_FP8_DENSE_GATED_SILU` | 1 | Fused `gate_up_proj` plus SiLU epilogue. Saves about 5.4 GiB per rank on a 27B TP2 case by avoiding duplicate layouts. |
| `VLLM_SM70_FP8_PREFILL_EXACT_DENSE` | see doc | Large-M exact dense prefill route. |
| `VLLM_SM70_NVFP4_TURBOMIND` | 1 | NVFP4 dense route. |
| `VLLM_SM70_SHARED_GATE_MAX_M` | 64 | M threshold for the shared gate path. |

The pairing of a default-on route with a named rollback switch is a pattern we
must copy. It makes every risky route reversible without a rebuild.

## 5. Attention for V100

`Tilelang-FA-V100` provides FlashAttention for SM70 in TileLang.

- `tilelang_fa_v100/_kernels_forward.py`, `_kernels_backward.py`
- `tilelang_fa_v100/_kernels_paged.py` and `_paged_adapter.py` for paged KV
- `tilelang_fa_v100/_decode_adapter.py` for decode
- `tilelang_fa_v100/_configs.py` for shape and tile selection
- Supports sliding window and head dimension 512

`1Cat-vLLM/flash-attention-v100/` is a separate CUDA FlashAttention port with
its own `kernel/` and `include/` directories.

`sglang-V100` runs with `--attention-backend tilelang_fa_v100`, so the TileLang
path is the one with published acceptance numbers.

## 6. Performance targets

From `sglang-V100/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md`.
The configuration is the real `Qwen3.8-27B-FP8` checkpoint, FP16 activations,
E5M2 KV, TP4 on four V100-SXM2-32GB with NVLink, one request at a time,
TileLang attention, and 256 generated tokens per point.

| Input | TTFT | Prefill | TPOT | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 342.256 ms | 2,991.9 tok/s | 17.180 ms | 58.21 tok/s |
| 4,096 | 990.204 ms | 4,136.5 tok/s | 17.356 ms | 57.62 tok/s |
| 25,000 | 6,731.904 ms | 3,713.7 tok/s | 19.703 ms | 50.75 tok/s |
| 70,000 | 23,488.233 ms | 2,980.2 tok/s | 25.760 ms | 38.82 tok/s |
| 128,000 | 54,322.900 ms | 2,356.3 tok/s | 33.289 ms | 30.04 tok/s |

Use the 1K and 4K rows as the first acceptance target. Decode falling with
context is a known unfinished item in that tree, so do not treat the long-context
decode numbers as a ceiling to beat.

`1Cat-vLLM/docs/design/sm70_fp8_long_prefill_exact_dense.md` records a separate
prefill improvement of 22.4 percent at 32K, 12.9 percent at 128K, and 7.3
percent at 256K, with output hashes preserved between control and candidate.

## 7. Format support matrix for SM70

Derived from the configs registered in our own tree.

| Format | Kernel file | Quant type | Works on SM70 | Weights per GPU at TP4 |
| --- | --- | --- | --- | ---: |
| FP16 or BF16 | `sm70_884_16.cu` | none | yes | 14.4 GiB |
| INT4 AWQ, K-grouped | `sm70_884_4.cu` | `kK` | yes, proven in production | 3.6 GiB |
| MXFP4 | `sm70_884_4.cu` | `kK` | yes | 3.6 GiB |
| FP8 E4M3, K-grouped | `sm70_884_8.cu` | `kK` | yes | 7.2 GiB |
| FP8 E4M3, block `{128,128}` | none directly | `kB` | only after scale expansion | 7.2 GiB |
| INT8 or W8A8 | none | none | no | not applicable |

### On INT8

INT8 is not a shortcut here. TurboMind has no INT8 weight format class in
`lmdeploy/turbomind/weight_format.py` and no INT8 operand configuration for
SM70. Choosing INT8 means writing both a format and a kernel. INT4 AWQ is
already supported, already proven on this cluster, and uses half the memory of
INT8, so INT8 has no remaining advantage.

## 8. Recommended order of work

1. Port the FP8 scale expansion into our loader so `FP8Format` produces
   K-grouped scales and declares `block_sizes` of `{128, 1}`. This makes
   `MakeQuantDesc` return `kK` and reaches the existing `Config_E4M3` tiles.
   No new kernel is required.
2. Re-apply the pre-SM90 packing change in `LinearWeight::prepare`. It is
   necessary but not sufficient on its own, and it was reverted earlier for
   that reason.
3. Add the eight missing E4M3 tiles from `1Cat-vLLM`, translating the
   registration syntax.
4. Adopt a named rollback switch for the FP8 route before enabling it by
   default.
5. Compare against the 1K and 4K rows in section 6.

## 9. Attribution

`sglang-V100` carries a commit titled `docs(v100): document upstream
attribution`. Any code we copy from these trees must carry the same
attribution discipline, and each copied block should name its source file and
the commit it came from.
