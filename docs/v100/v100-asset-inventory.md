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
| `ds4/kernels/` | — | Our own earlier V100 work. Holds the only FP16-accumulate `m8n8k4` wrapper found in any tree. |

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

## 9. Our own ds4 kernels

These are in `ds4/kernels/`, copied from a DeepSeek and llama.cpp working tree
at commit `5903432d826b7b10cdc6d02d8d5da1bbe65371b8`. They are our own prior
work, so there is no third-party licensing question, and `mma_sm70.cuh` is
itself adapted from TurboMind's `core/mma.h`.

### 9.1 The FP16-accumulate MMA wrapper

This is the highest-value item in `ds4`, because no other tree has it.

File: `ds4/kernels/tc-grid/kernels/mma_sm70.cuh`, line 103.

```text
mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16
```

Every `m8n8k4` wrapper in our TurboMind and in the vendored copy inside
`1Cat-vLLM` uses the FP32 accumulator form only:

| Tree | `m8n8k4` accumulate forms present |
|---|---|
| `lmdeploy/src/turbomind/kernels/core/mma.h` | `f32` only |
| `1Cat-vLLM` vendored `core/mma.h` | `f32` only |
| `ds4/kernels/tc-grid/kernels/mma_sm70.cuh` | `f32` and `f16` |

Both trees do contain `f16.f16.f16.f16` strings, but only for `m16n8k8` and
`m16n8k16`, which are SM75 and later shapes. For the SM70 `m8n8k4` shape, the
FP16 accumulator wrapper exists only in `ds4`.

The source comments claim the FP16 accumulator doubles the tensor-core peak,
from about 62 TFLOPS to about 125 TFLOPS, and halves the accumulator register
footprint from 32 bytes to 16 bytes per atom fragment. Treat both numbers as
unverified on our hardware until measured.

### Do not accumulate in FP16 for this model

The recorded tolerance is `rel <= 1e-3` and `p99 <= 0.05` and `maxabs <= 0.1`.
That band was accepted for a GEMM microbenchmark. It is not sufficient evidence
for a BF16 base model, and the FP16 accumulator must not be adopted as written.

The checkpoint is BF16. Every tensor sampled from
`/srv/models/Qwen3.8-27B-FP8` reports dtype `BF16`. BF16 carries 8 mantissa
bits with the exponent range of FP32. FP16 carries 11 mantissa bits with a
maximum finite value of 65504. Swapping an FP32 accumulator for an FP16 one
therefore trades away exponent range that the model was trained to use.

Two distinct failure modes apply, and only the first is loud:

1. Overflow. A partial sum that reaches 65504 becomes infinity, and the value
   never recovers.
2. Swamping. Once the accumulator is large, small addends round away. With
   `eps = 2^-11`, an accumulator near 1024 silently discards every addend
   below 0.5.

Swamping is the dangerous one. It produces no error and no warning. It removes
the small contributions that a long reduction is supposed to sum, and the
visible result is a model that is quietly worse rather than a model that fails.

The reduction lengths here are long enough for this to matter:

| Projection | K at TP4 | Worst-case `K * eps` | Random-walk `sqrt(K) * eps` |
|---|---:|---:|---:|
| `gate_up_proj` | 5120 | 2.50 | 0.035 |
| `down_proj` | 4352 | 2.12 | 0.032 |
| `down_proj`, unsharded K | 17408 | 8.50 | 0.064 |

Even the optimistic random-walk column sits far above the `rel <= 1e-3` figure
that the microbenchmark accepted.

The accumulator is not reset inside the K-loop. In both `v12_kernels.cuh` and
`v13_kernels.cuh`, `c_frag` is declared `half[ATOMS_M][ATOMS_N][8]`, zeroed
once, and accumulated across the whole reduction. The full K chain runs in
FP16.

A microbenchmark also cannot see what we care about. Elementwise tolerance on
one GEMM says nothing about accumulated degradation across 48 layers, and
nothing about the long-context behavior where swamping compounds. The honest
acceptance test is end-to-end output quality, not a per-kernel error bound.

### A possible adaptation

The idea is salvageable if the FP16 chain is kept short and the outer reduction
stays in FP32. The SplitK kernel already does exactly this. When `KS > 1`,
`v12s` reduces partial results with an FP32 `atomicAdd` in global memory:

```cpp
// v12_kernels.cuh:519
//   - When KS  > 1: atomicAdd to gmem C. Caller MUST pre-zero C.
atomicAdd(&C[(size_t) gm * N + gn + 0], f.x);
```

At `KS = 8`, each FP16 chain covers only `K / 8`:

| Projection | K | Per-chain length at `KS = 8` |
|---|---:|---:|
| `gate_up_proj` | 5120 | 640 |
| `down_proj` | 4352 | 544 |
| `down_proj`, unsharded K | 17408 | 2176 |

That is the shape of a safe adaptation: FP16 inside a short chain, FP32 across
chains. It is still not proven for a BF16 model, and it costs some of the
speedup, because the FP32 atomic reduction is extra traffic.

A second option is to keep the FP32 accumulator and take only the register
saving, which is unavailable, because the register saving comes from the FP16
accumulator itself.

### Required gate before any adoption

If this is attempted, it must be behind a named switch that defaults to off,
and it must clear end-to-end quality evidence rather than a kernel tolerance
band. At minimum, compare greedy-decode outputs against the FP32-accumulator
build on a fixed prompt set, at short and long context, and treat any
systematic divergence as a failure. Do not accept a per-element error bound as
a substitute.

### 9.2 INT8 tensor-core GEMM for SM70

This revises the INT8 conclusion in section 7. Section 7 says TurboMind has no
INT8 format and no SM70 INT8 kernel, which remains true of TurboMind. It is not
true of our own tree.

| File | Lines | Content |
|---|---:|---|
| `tc-grid/kernels/v13_kernels.cuh` | 2090 | Eight INT8 GEMM variants, `v13_rf` through `v13_rf_v8`. |
| `tc-grid/kernels/v12_kernels.cuh` | 719 | `mm_int8_lut_v12`, `_v12_ms3`, and the SplitK `v12s`. |
| `tc-grid/include/dispatch.h` | 130 | Per-M champion table and `choose_kernel(M, N, K)`. |
| `tc-grid/include/tc_grid.h` | 145 | Format and unpack-path enums, block sizes, harness types. |

The design line is documented and specific. `v13_rf` stores raw INT8 in shared
memory at 1 byte per weight and dequantizes with PRMT in registers between the
load and the MMA, instead of round-tripping dequantized FP16 through shared
memory at 2 bytes per weight. The comment states this matches what TurboMind's
`Transform_HMMA_SIMT_B` already does inside `SmemCopyAtom_Pack_v3`.

Recorded results, all unverified by us:

- `v13_rf_v6` beats `v12_ms3` by 27 percent at M=2048, N=K=7168, and reaches a
  50 TFLOPS goal.
- It wins on 23 of 24 asymmetric shapes, with a best case of 177 percent at
  M=256, N=7168, K=18944.
- The starting point was 31.7 percent HMMA-active against 56 percent for
  TurboMind FP8 at the same shape and MMA family.

The dispatcher encodes the champions:

```text
M in [1, 128)    -> v12s_64x128x32_w4_ks8       (SplitK, KS=8)
M in [128, 512)  -> v13_rf_v6_128x128x16_w4
M >= 512         -> v13_rf_v6_128x128x16_w4
```

`dispatch.h` was written to be shared between the lab harness and a TurboMind
runtime integration, and it names `LlamaLinear.cu` and `moe_ffn_layer.cc` as
the intended runtime callers. That integration wrapper does not exist in our
tree yet.

### 9.3 Standalone V100 kernels in ds4/kernels/v100

A separate set of header-only kernels, 3442 lines in total.

| File | Lines | Notable kernels |
|---|---:|---|
| `attention.cuh` | 1002 | `attention_raw_swa_window_kernel`, `kv_fp8_round_store_raw_swa_kernel`, `rope_tail_rows_kernel`. Sliding-window attention with an FP8 KV store. |
| `router.cuh` | 429 | `router_logits_ep_from_rank_major_kernel`, `router_logits_allreduce_partial_kernel`, `shard_top1_kernel`. MoE routing with expert parallelism. |
| `norm.cuh` | 229 | `rms_norm_plain_rows_stable_kernel`, `head_rms_norm_local_heads_kernel`. |
| `hc_mix.cuh` | 225 | Weighted-sum and shard-mixing kernels. |
| `dense.cuh` | 185 | `f8_b128_dense_kernel` and `f8_b128_dense_hmma_m16_kernel`. |

### 9.4 Why dense.cuh does not solve the FP8 problem

`f8_b128_dense_kernel` looks like an exact match for our format and is not one.
It assumes a GGUF-style interleaved layout, where one E8M0 scale byte is
followed by 128 E4M3 data bytes in a 129-byte stride:

```cpp
const uint8_t *block = wrow + (uint64_t)(c / 128u) * 129ull;
const float scale = f8_e8m0_to_f32_dev(block[0]);
const float w = f8_e4m3fn_to_f32_dev(block[1u + (c % 128u)]) * scale;
```

Our checkpoint stores a separate `weight_scale_inv` tensor in BF16, with one
scale per 128-by-128 tile, and the weight bytes are contiguous. The scale type,
the blocking, and the memory layout all differ. The mathematical idea is the
same, the data layout is not, so this kernel is a reference for the arithmetic
rather than a drop-in path.

Its second variant uses the `wmma` API with 16-by-16-by-16 tiles and an FP32
accumulator, which is a different and less tuned path than the `m8n8k4`
intrinsics used in `tc-grid`.

### 9.5 How to use ds4 in this project

Ranked by expected value against effort.

1. Treat the FP16-accumulate `m8n8k4` wrapper as a research item, not a
   scheduled task. It is the only unique asset here, and it is also the most
   dangerous one. Do not adopt it as written. If it is attempted, adapt it so
   the FP16 chain stays short and the outer reduction is FP32, gate it behind
   a default-off switch, and qualify it on end-to-end output quality. The
   recorded kernel tolerance band is not sufficient evidence. See section 9.1.
2. Reuse the `dispatch.h` per-M champion pattern. Our current FP8 dispatch has
   no M-direction routing at all, and small-M decode against large-M prefill is
   exactly the split that table encodes.
3. Keep the INT8 stack as evidence, not as a plan. It shows INT8 on SM70 is
   achievable and roughly what it costs. It does not change the section 7
   recommendation, because INT4 AWQ is already supported end to end and uses
   half the memory.
4. Treat `attention.cuh` and `router.cuh` as alternates to the TileLang path,
   worth reading only if TileLang attention does not integrate cleanly.

## 10. Attribution

`sglang-V100` carries a commit titled `docs(v100): document upstream
attribution`. Any code we copy from these trees must carry the same
attribution discipline, and each copied block should name its source file and
the commit it came from.
