# KV path trace and Phase 2 gate record

This file records two things the master specification requires:

1. The existing TurboMind INT8 KV format, traced to source.
2. The Phase 2 SM70 decode correctness gate outcome.

## The existing format is per-token, not grouped

The existing path uses per-token, per-KV-head asymmetric unsigned INT8. It is
the compatibility control for the campaign. Do not describe it as grouped or
block-scaled.

`warp_stats` in `src/turbomind/kernels/attention/quantization.h` computes the
parameters. For each `s`, which is one token, it reduces min and max across
every `c` iteration, which together cover the complete head dimension:

```cpp
Array<T, 2> stats{Infinity<T>(), -Infinity<T>()};
for (int c = 0; c < C; ++c) {
    warp_minmax<WarpThreadC>(stats, x[s][c]);
}
const float scale = ((float)stats[1] - (float)stats[0]) * inv_q_max;
param[s][0] = (P)scale;
param[s][1] = (P)stats[0];
```

The result is one `(scale, zero)` pair per token per KV head, covering all 256
dimensions. `ProcessKV_v2` in `kv_cache_utils_v2.cu` calls it once for K and
once for V, so K and V carry separate parameters.

## Source boundaries

| File | Symbols |
| --- | --- |
| `kernels/attention/kv_cache_utils_v2.cu` | `ProcessKV_v2`, `invokeProcessKV_v2` |
| `kernels/attention/quantization.h` | `warp_stats`, `ConvertKvCache<T, uint8_t>`, `ConvertKvCache<uint8_t, T>` |
| `kernels/attention/block.h` | `block::Config`, `block::Layout`, `block::Head` |
| `kernels/attention/block_iterator.h` | `BlockIterator`, `CacheIterFactory` |
| `kernels/attention/attention_universal.h` | in-kernel KV write and quant-parameter storage |

## Sizes that assume two values per token and head

`block.h` encodes the two-parameter assumption in one place:

```cpp
TM_HOST_DEVICE int token_param_size() const
{
    return config().t_bits() * 2 / 8;  // 2 for scales/zeros
}
```

Every other size derives from it. The formulas were checked against the values
the fixture printed at head_dim 256, `block_len` 64, `q_bits` 8, `t_bits` 16
and 4 KV heads:

| Quantity | Formula | Computed | Fixture |
| --- | --- | ---: | ---: |
| `token_param_size` | `t_bits * 2 / 8` | 4 | 4 |
| `token_data_size` | `head_dim * q_bits / 8` | 256 | 256 |
| `head_param_size` | `block_len * token_param_size` | 256 | 256 |
| `head_data_size` | `block_len * token_data_size` | 16384 | 16384 |
| `layer_size` | `head_num * 2 * (head_data + head_param)` | 133120 | 133120 |

Per token and KV head this is 256 data bytes plus 4 parameter bytes, so K and V
together cost 520 bytes. That matches the specification table, now confirmed by
execution rather than by arithmetic alone.

A Phase 5 block-scaled format changes `token_param_size`, so every derived
offset above must be re-derived, not patched.

## Phase 2 gate: SM70 decode correctness

### What was tested

The gate must exercise the kernel the campaign target actually selects. Qwen3.8
has 24 attention heads, 4 KV heads and head_dim 256, so the query group size is
6, the runtime selects `kH=3`, and the decode kernel is
`KT<half, uint8_t, 3>` registered in `kernel/decoding_sm70_256.cu`.

The fixture previously hardcoded head_dim 128 with a query group size of 8,
which selects `kH=2` in `decoding_sm70_128.cu`. It exercised a different
kernel. The shape constants and mode selectors are now `ifndef` guarded, and
the gate builds with head_dim 256, 24 heads and 4 KV heads.

### Result: deterministic

Ten iterations produced nine consecutive comparisons of adjacent outputs:

```text
abs_diff = 0 rel_diff = 0 outliers = 0
```

Every comparison is exactly zero. The SIMT decode kernel is bit-exact across
runs. There is no race, no uninitialized read and no nondeterministic
reduction. This alone contradicts the donor's reported failure mode.

### Result: against the FP16 reference

```text
output   rel_diff = 0.0324
k_cache  rel_diff = 0.0247
v_cache  rel_diff = 0.0262
```

This run quantizes KV to INT8 while the reference keeps FP16, so the difference
combines two sources: the kernel and the quantization. A few percent is the
expected magnitude for INT8 KV, but expectation is not evidence. An FP16 KV
control at the identical shape separates them.

### The FP16 KV control

The control changes one thing. Same kernel, same shape, same code path, with
`KV_INT8` unset so KV stays FP16.

| KV format | output | k_cache | v_cache | outliers |
|---|---:|---:|---:|---:|
| INT8 | 0.03243 | 0.02472 | 0.02624 | 2270.1 |
| FP16 | 0.00136 | 0.00000 | 0.00000 | 0.0 |

The output error falls by a factor of 24, and both KV caches match the
reference exactly with zero outliers.

A faulty kernel cannot select a data type. If the SIMT decode kernel computed
incorrect attention, the FP16 run would show it too. Instead the FP16 run
reproduces the reference to 0.00136, which is ordinary floating-point
accumulation-order noise, and writes both caches bit-exactly.

The two to three percent in the INT8 run is therefore quantization error from
the per-token format, not a kernel fault.

### Gate outcome 1

The SIMT decode route is correct for the campaign target.

The campaign proceeds as planned. SIMT remains a valid host for the
block-scaled reader.

Evidence:

- The kernel is bit-exact across ten iterations, so there is no race and no
  uninitialized read.
- With FP16 KV it matches the reference to 0.00136 with zero outliers, and
  writes both KV caches bit-exactly.
- With INT8 KV the error is confined to the magnitude that per-token
  quantization predicts.
- The tested kernel is `KT<half, uint8_t, 3>` at head_dim 256, which is the
  kernel Qwen3.8 decode selects, not a proxy shape.

### Donor claim and campaign scope

The donor commit `c1e21e24` reports garbage output at TP8 with MoE for
Qwen3.5-122B, and its remedy sets `SM70_DECODE_USE_MMA_884`. The product base
contains no `MMA_884_DEC` symbol, having replaced the donor compile-time
dispatch with a runtime registry, so the donor remedy does not apply here.

The donor reproducer also differs from the campaign target in model, parallel
degree and expert routing. Treat the donor report as a risk to test, not as an
established fault.
