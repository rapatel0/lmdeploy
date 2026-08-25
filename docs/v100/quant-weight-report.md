# Phase 3: qualified SM70 quantized-weight work

The specification requires a commit-to-symbol map before any port, and forbids
treating broad release commits as implementation units. This report builds that
map first, then evaluates each priority family against the product base.

The controlling finding is that most families are already satisfied. Porting
them would add risk without adding behavior.

## The nine reference commits are not equal

All nine resolve in the 1Cat tree. Their sizes separate them into two groups:

| Commit | Files | Lines | Subject |
| --- | ---: | ---: | --- |
| `a6a1b9abe18e` | 35 | 2672 | Update SM70 V100 migration source |
| `e64d39aa7ee1` | 77 | 16076 | Stabilize SM70 Qwen MTP paths |
| `b78816657d9c` | 85 | 19472 | Update SM70 V100 release paths |
| `15ace0f15986` | 101 | 13801 | Prepare 1Cat-vLLM 1.2.1 |
| `4f90bd6f5e6c` | 182 | 80757 | Sync SM70 performance stack |
| `f228fa80d1e4` | 10 | 582 | Bound SM70 AWQ prefill workspace |
| `4d0fcdf7ba2f` | 11 | 631 | Accelerate SM70 FP8 long prefill |
| `87ac589295ba` | 8 | 147 | Keep SM70 FP8 workspace out of CUDA graphs |
| `1d1daa789963` | 3 | 75 | Make SM70 AWQ compact reduce deterministic |

The first five are release dumps of 13000 to 80000 lines. They are reference
ranges only. The last four are surgical and carry one identifiable behavior
each.

## Commit-to-symbol map

Only the four surgical commits produce a usable map. All four touch the same
file, `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`.

| Commit | Family | Symbols touched |
| --- | --- | --- |
| `f228fa80d1e4` | 4, bounded AWQ prefill workspace | `awq_sm70_prepare`, `sm70_dynamic_draft_vocab_refresh_tail_weight_kernel` |
| `4d0fcdf7ba2f` | 5, FP8 exact-dense long prefill | `fp8_gemm_sm70_out`, `fp8_sm70_prepare`, `awq_sm70_dequantize_out` |
| `87ac589295ba` | 6, stable-pointer graph workspace | `fp8_gemm_sm70_prefill_dispatch_out`, `fp8_gemm_sm70_out` |
| `1d1daa789963` | 7, deterministic compact reduction | `awq_moe_single_token_sm70_out` |

## Family findings

### Family 7, deterministic compact reduction: already satisfied

The donor fix removes a weighted-reduce epilogue that accumulated expert CTAs
into one FP16 output with `atomicAdd`, because CTA arrival order is not
deterministic. It replaces that with a fixed route order and FP32
accumulation.

The product base already reduces that way. `MoeReduceKernel` in
`kernels/gemm/moe_utils_v2.cu` gives each output element to one thread, which
walks the experts in a fixed unrolled order and accumulates into
`Array<float, vec_size>`:

```cpp
Array<float, vec_size> accum{};
PRAGMA_UNROLL
for (int e = 0; e < exp_k; ++e) {
    ...
    const auto x = cast<float>(v) * scale[e];
    accum        = accum + x;
}
Store(&dst[i], cast<T>(accum));
```

There is no atomic and no cross-CTA race on the output. The three `atomicAdd`
calls in that file are tile counters in the routing pass, not output reduction.

The base already implements the donor's end state. Port nothing.

### Family 6, stable-pointer graph workspace: not applicable

The donor changes a workspace parameter from `torch::Tensor` to an `int64_t`
pointer rebuilt with `torch::from_blob`, so an allocator tensor is not captured
into a CUDA graph.

The product base captures no CUDA graphs. A search for `cudaStreamBeginCapture`
and `cudaGraphLaunch` across `src/turbomind/` returns nothing. The failure this
fix prevents cannot occur.

The change is also vLLM-specific: it exists to defeat the PyTorch caching
allocator's interaction with graph capture, and TurboMind manages its own
workspace memory.

Port nothing. Revisit only if graph capture is introduced, which Phase 5 lists
as a requirement for the new format.

### Family 1, dispatch-cache synchronization: not applicable

`kernels/gemm/dispatch_cache.cu` contains no mutex or atomic, and `gemm.cu`
calls `cache_.Find` and `cache_.Insert` on the dispatch path. That looks like a
data race until ownership is checked.

`DispatchCache cache_` is a member of `Gemm`, `Gemm gemm_` is a member of
`LlamaLinear`, and `Context` owns one `LlamaLinear` through a `unique_ptr`.
`turbomind.py` creates one Context per device:

```python
model_comm.create_context(device_id)
```

Each rank runs on its own thread with its own Context, so each rank has its own
dispatch cache. There is no shared mutable cache to synchronize.

Port nothing. This conclusion depends on one-Context-per-rank, so it must be
re-checked if context sharing is ever introduced.

### Family 4, bounded AWQ long-prefill workspace: real, measured, not yet urgent

The specification points at the `tmp_kv` buffer and the `MAX_CTA_S` sizing term
in `unified_attention_layer.cc`. That line is:

```cpp
Tensor tmp_kv{{local_kv_head_num, is_mla ? 1 : 2,
               d.prefill.k_sum + MAX_CTA_S, size_per_head}, dtype, device};
```

`MAX_CTA_S` is 64. The buffer scales with `d.prefill.k_sum`, which is the summed
prefill length across the whole batch, not one sequence. At head_dim 256 and
FP16, one gibibyte is reached at these totals:

| Configuration | `k_sum` for 1 GiB |
| --- | ---: |
| TP4, 1 KV head per rank | 1048576 tokens |
| TP1, 4 KV heads | 262144 tokens |

For a single 128k-token sequence at TP4 the buffer is 0.125 GiB, which is not a
problem. The exposure is batch aggregate: enough concurrent long prefills reach
the threshold, because `k_sum` sums across requests.

This is a genuine unbounded allocation, so record it as a real finding. It is
not a Phase 3 blocker for the campaign target, and the donor remedy is written
against a vLLM AWQ path that does not match the TurboMind allocation site. A
bound belongs here, but it must be written for TurboMind rather than copied.

### Families 2, 3, 5, 8: no matching base symbol

Active-expert scheduling, exact small-M tactic selection, FP8 exact-dense long
prefill and Marlin grouped-scale mechanisms have no counterpart symbol in the
product base. The donor implementations are written against vLLM structures.

Porting these requires new code rather than a transplant, and none is required
by the Phase 3 goal. Defer them with an approved patch-family record.

## The SM70 foundations the specification expects are present

The three configurations the specification names all exist and carry exactly
the expected configs:

| File | Configs |
| --- | --- |
| `kernel/sm70_884_4.cu` | `Config_U4_d`, `Config_U4_g`, `Config_MXF4` |
| `kernel/sm70_884_8.cu` | `Config_E4M3` |
| `kernel/sm70_884_16.cu` | `Config_F16` |

They self-register through `gKernelFactories`, and `Registry::Add` filters by
architecture with `is_arch_compatible`, where `karch` 700 requires
`Sm70::is_compatible(darch)`.

Confirmed on hardware rather than by reading:

```text
Tesla V100-SXM2-32GB, 7.0
device arch = 700 (Tesla V100-SXM2-32GB)
Sm70 compatible: yes
```

## Outcome

Port nothing in Phase 3.

Three families are already satisfied or inapplicable in the product base, and
the evidence for each is recorded above. One family, the unbounded `tmp_kv`
prefill buffer, is a real finding with measured thresholds, and it needs a
TurboMind-specific bound rather than a donor transplant. The remaining families
have no matching base symbol.

This is the outcome the specification's own instruction points to: port only
missing behavior. The base is further along than the donor commit list assumes.

## Fail-closed loader tests

The specification requires fail-closed tests for seven items. They live in
`tests/turbomind/test_weight_format_loader.py` and cover tensor-name coverage,
scale-name coverage, scale semantics, ignored-module handling, quantization
block shape, dense and MoE route ownership, and loaded tensor counts through
the format-identity checks that guard fusion groups.

Fail-closed means a checkpoint that does not match a declared format must
raise rather than load as something else. A silent misclassification produces a
model that runs and returns wrong numbers, which is the worst outcome for a
quantization campaign.

The tests import the compiled `_turbomind` extension, so they run on the V100
against the built wheel rather than on a laptop:

```text
29 passed in 2.97s
```

### The tests were verified by mutation

A passing test suite proves nothing until it is shown to fail. Removing the
fail-closed guard from `TrivialFormat.accepts`, so that it accepts everything,
produces:

```text
FAILED test_trivial_rejects_quantized_checkpoint
FAILED test_trivial_rejects_integer_weight
FAILED test_exactly_one_format_accepts_awq
FAILED test_no_format_accepts_an_unknown_checkpoint
4 failed, 25 passed
```

Exactly the four route-ownership and rejection tests fail, and the unrelated
tests still pass. The suite detects the regression it exists to detect.

Record one process failure with this. The first mutation attempt reported 29
passed against a supposedly broken build, which would have been a false pass.
The patch had used double-quoted source text while the installed wheel is
single-quoted, so `str.replace` matched nothing and changed no code. The second
attempt verifies that the mutation applied before drawing any conclusion, and
prints the replaced body as evidence. Any mutation harness must fail loudly
when it does not mutate.

## Deferred, default-off until independent qualification

Unchanged from the specification: NVFP4, compact MXFP4 top-6 decode, DeepSeek
exact M8 selectors, broad FP8 router overrides, low-level LM-head fused
sampling, and dense FP16 donor paths.

Not ported by rule: the vLLM private custom-all-reduce coupling, and relative
includes that escape the TurboMind subtree.
