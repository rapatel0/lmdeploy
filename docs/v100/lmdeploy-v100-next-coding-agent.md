# Coding-Agent Master Specification: Build `lmdeploy-v100-next`

## Execution contract

Treat this document as the campaign master specification.

The campaign outcome is block-scaled INT8 KV, default-on for Qwen3.8 on V100.

FP16 KV is the starting baseline, not a deliverable.

Read the primary objective before any phase.

Do not execute all phases in one autonomous run.

Run Phase 0 and Phase 1 first.

Stop for operator approval after the source lock and audit reports exist.

Stop again before these actions:

- Transfer donor code.
- Reserve all eight V100 GPUs.
- Change a default runtime path.
- Promote a container or homelab deployment.
- Re-lock any source after the first approval.

Record every operator approval in `docs/v100/approvals.md`.

Record the date, the approved scope, and the artifact SHA-256 digest.

Give each phase a timebox and an abort rule before the phase starts.

Stop and report when a phase exceeds its timebox.

## Role

Act as the lead CUDA and inference-runtime engineer for `lmdeploy-v100-next`.

Build a V100-focused LMDeploy fork from current upstream LMDeploy.

Use semantic ports from the audited donor repositories.

Do not overlay an old source tree onto current LMDeploy.

TurboMind is the only engine in scope.

The PyTorch engine keeps upstream behavior without change.

## Workspace boundary

Use this workspace root:

```text
~/repos/cust-lmdeploy/
```

Use these local repositories:

```text
~/repos/cust-lmdeploy/lmdeploy/          product repository
~/repos/cust-lmdeploy/1Cat-vLLM/        SM70 runtime donor
~/repos/cust-lmdeploy/sglang-V100/      SM70 attention donor
~/repos/cust-lmdeploy/Tilelang-FA-V100/ standalone TileLang attention reference
~/repos/cust-lmdeploy/marlin_v100/      SM70 quantized GEMM reference
~/repos/cust-lmdeploy/lmdeploy-v100/    read-only correction donor, created in Phase 0A
```

Treat `Tilelang-FA-V100` and `marlin_v100` as reference-only checkouts.

Do not copy either source tree or add either repository as a product submodule.

Port only approved symbols or mechanisms after the provenance, license, and compatibility audit.

Neither reference checkout contains a LICENSE file at its root.

The audit fails closed on an unknown license, so both fail today.

Resolve the license state for both repositories before Phase 1.

If a repository has no resolvable license, drop it from the campaign.

Use `~/repos/cust-lmdeploy/lmdeploy/` as the only LMDeploy source repository.

Create the campaign branch from the approved source lock:

```text
experiment/lmdeploy-v100-next
```

Do not edit donor repositories.

Do not add LMDeploy source files to `~/repos/homelab/`.

Put Kubernetes manifests and cluster runbooks in this homelab path only:

```text
~/repos/homelab/manifests/lmdeploy-v100-next/
```

Keep large build outputs, model files, CUDA traces, and profiler reports outside Git.

Use this node-local artifact root:

```text
/localpool/lmdeploy-v100-next/
```

Record each retained artifact path and SHA-256 digest in the qualification report.

## Homelab hardware allocation

Use `gpu-01` for all SM70 tests.

`gpu-01` contains eight Tesla V100-SXM2-32GB GPUs in two four-GPU NVLink islands.

At the current checkpoint, the active TP4 service uses host GPUs 0-3.

Host GPUs 4-7 form the free NVLink island.

Use the free island for TP1 and TP4 work without a production outage.

Before each GPU run, verify all of these facts:

- The active workload and its GPU UUIDs.
- The free island and its GPU UUIDs.
- The `nvidia-smi topo -m` output.
- The Kubernetes device assignment.
- The CPU and NUMA affinity.
- The absence of stale compute processes.

Do not rely on the recorded host indexes after a workload change.

Require operator approval before TP8 tests across both islands.

Do not alter the active SGLang or 1Cat workspace.

Do not force-delete GPU pods.

Use normal pod termination to avoid leaked CUDA worker processes.

## Toolchain lock

Build SM70 code with CUDA 12.8.1.

Do not use CUDA 13 NVCC for V100 code.

Current LMDeploy omits `70-real` when the CUDA compiler version is 13 or later.

Use these initial toolchain constraints:

```text
build image: nvidia/cuda:12.8.1-devel-ubuntu24.04
runtime family: CUDA 12
CMake CUDA architecture: 70-real
Python: 3.12
PyTorch: the CUDA 12 build approved in the source lock
```

Pin the exact image digest, compiler versions, dependency lock, and PyTorch wheel.

Verify that each retained CUDA library contains SM70 machine code.

Do not accept a PTX-only SM70 build.

## Source resolution and immutable lock

### Product base

Use this single product base:

```text
InternLM/lmdeploy tag v0.16.0
1208bf006bbac69f1f012ceafeeeb70f623b632c
```

A tag gives a stable base and a clear upstream diff for the campaign.

Do not use the observed main commit as the product base.

Record the observed main commit as a drift reference only.

The local checkout head is not the product base.

At the current checkpoint the local head is `b56ddfb6`.

That head is 73 commits behind tag `v0.16.0`.

The `origin` remote is the fork `git@github.com:rapatel0/lmdeploy.git`.

The `upstream` remote is `git@github.com:InternLM/lmdeploy.git`.

A fetch of `origin` alone does not advance the tree.

Fetch `upstream` and check out the product base before any other work.

### Audit anchors

Use these revisions as the initial audit anchors:

```text
InternLM/lmdeploy tag v0.16.0
1208bf006bbac69f1f012ceafeeeb70f623b632c

InternLM/lmdeploy drift reference, observed main
1263d5cd889b7ad500e85b8eb20401327a4516ab

zh-nj/lmdeploy-v100
d7c29f88e44016d7a757850fe761c1b1b66181c8

1CatAI/1Cat-vLLM
675a12dedcca8cc020c033f1b1d0f1751d4b8efe

haohervchb/sglang-V100
0083b9fd83a601b1fcd9a691f7240be4e6be111e

haohervchb/Tilelang-FA-V100
c6332ebb0670efb7702eeb1b0e0d4477bea49def

zhinianqin/marlin_v100
3f16d442cdb4c24dd225bbec196c982a54d9a31c
```

The observed remote heads already differ from some local checkout heads.

Fetch every remote before implementation.

Write this immutable lock file:

```text
docs/v100/source-lock.json
```

Record these values for every source:

- Repository URL.
- Requested ref.
- Resolved commit SHA.
- Commit date.
- Tree SHA.
- License file SHA-256, or an explicit `no-license-file` result.
- SPDX license identifier.
- Local checkout path.
- Local dirty-state result.

At the current checkpoint, three sources carry an Apache-2.0 LICENSE file:

```text
lmdeploy
1Cat-vLLM
sglang-V100
```

Two sources carry no LICENSE file at their root:

```text
Tilelang-FA-V100
marlin_v100
```

Record the `no-license-file` result for both, then resolve it before Phase 1.

Record new remote heads separately from the approved audit anchors.

Do not change the source lock after operator approval.

No local checkout of `zh-nj/lmdeploy-v100` exists at the current checkpoint.

Phase 1 source B and all of Phase 2 require it.

Create a read-only checkout at `~/repos/cust-lmdeploy/lmdeploy-v100/` in Phase 0A.

### Upstream drift and re-lock

The campaign spans upstream releases.

Do not re-lock a source inside an approved phase.

Request a re-lock only at a phase boundary.

A re-lock request must state these items:

- The new resolved commit.
- The upstream diff for every file the campaign touched.
- The gates that a rebase invalidates.
- The cost to re-qualify those gates.

After an approved re-lock, re-run every invalidated gate before further work.

## Primary objective

The campaign outcome is block-scaled INT8 KV, qualified and default-on for Qwen3.8 on V100.

Everything before Phase 5 exists to make that outcome measurable and safe.

FP16 is the starting point, not a deliverable.

### Outcome

Deliver block-scaled INT8 KV with all of these properties:

1. Quality within the declared tolerance of the FP16 KV baseline.
2. About half the KV bytes of FP16 KV.
3. Better quality than the existing per-token INT8 KV format at equal bytes.
4. A qualified long-context gain that FP16 KV cannot reach in the same memory.
5. Graph-safe execution on TP4, and on TP8 after operator approval.

### Supporting work

Each item below is a prerequisite for the outcome, not an independent goal:

1. Preserve current LMDeploy model and serving behavior.
2. Establish the FP16 KV reference on the small model at TP1.
3. Establish the FP16 KV reference on Qwen3.8 at TP4.
4. Qualify SM70 AWQ, FP8, and MXFP4 weight paths, so the KV study runs on the production weight path.
5. Select the SM70 attention kernel that will host the block-scaled reader.
6. Preserve graph-safe long-context execution.

### Why FP16 KV is only a baseline

FP16 KV costs 64 KiB for each token of Qwen3.8 full attention.

At long context that cost, not compute, sets the concurrency limit.

See the KV memory budget section for the computation.

Block-scaled INT8 halves that cost.

The campaign succeeds when the halved cost holds quality.

The campaign fails when only FP16 KV qualifies.

### Scope limits

Do not qualify Qwen3.8 on TP1.

The FP8 weights need about 27 GB and a V100-SXM2 has 32 GB.

TP1 leaves no usable KV budget for that model.

TurboMind MTP is not part of this campaign.

See the separate MTP campaign section.

Do not add DFlash2 during the initial campaign.

Do not change checkpoint weight bytes at any point in the campaign.

Do not change the existing TurboMind KV format at any point in the campaign.

Keep that format as the compatibility control.

Do not claim support for every upstream model.

Preserve upstream behavior outside the closed qualification matrix.

## Closed qualification matrix

Freeze the complete model records before GPU work.

Use this target model:

```text
model repository: unsloth/Qwen3.8-27B-FP8
local model path: /srv/models/Qwen3.8-27B-FP8
architecture: Qwen3_5ForConditionalGeneration
layers: 64 total, 48 Gated DeltaNet and 16 full attention
full attention: 24 Q heads, 4 KV heads, head_dim 256
linear attention: K and V head_dim 128
native context: 262144
weight format: FP8 weight-only with FP16 compute on SM70
activation dtype: FP16
```

Record the exact model revision, tokenizer revision, configuration hash, index hash, and weight-file hashes.

Record the chat template and generation configuration.

Select one small upstream-supported FP16 model for baseline isolation.

Name that model in this document before the first run.

Record its repository, revision, tokenizer revision, and weight hashes.

Select a model that fits one V100 at FP16 with room for KV.

Use the small model for TP1 kernel and serving isolation.

Use Qwen3.8 for the production-shape, GDN, D256 GQA, and long-context gates.

Do not use “Qwen3.5 or Qwen3.8” as an interchangeable test target.

### KV memory budget

Compute the KV budget before any long-context run.

Qwen3.8 full attention uses 16 layers, 4 KV heads, and head_dim 256.

FP16 KV costs 64 KiB for each token:

```text
16 layers x 4 KV heads x 256 dims x 2 tensors x 2 bytes = 65536 bytes
```

At the native 262144-token context that is 16 GiB of KV for one sequence.

That figure excludes the GDN recurrent state and block fragmentation.

Record a budget table in `docs/v100/baseline.md` with these columns:

- TP degree.
- KV format.
- Weight bytes per rank.
- Aggregate KV bytes available.
- Maximum context at one concurrent sequence.
- Maximum concurrent sequences at 32K context.

Fill one row for each KV format:

```text
FP16 KV                            1024 bytes per token and KV head
existing per-token INT8 KV          520 bytes per token and KV head
block-scaled INT8 KV                528 bytes per token and KV head
```

That table states the campaign target in capacity terms.

The block-scaled row is the projected outcome before Phase 5 measures it.

Compare the projected row against the measured row in the Phase 5 report.

The model has 4 KV heads and TurboMind repeats KV heads to a TP-divisible count.

See `lmdeploy/turbomind/builders/attention.py`, function `repeat_kv_for_tp`.

At TP8 the 4 heads become 8, so aggregate KV memory doubles against TP4.

Record that doubling in the budget table.

Fail a context gate early when the budget does not fit.

## Required product repository structure

Create these artifacts inside `~/repos/cust-lmdeploy/lmdeploy/`:

```text
docs/v100/source-lock.json
docs/v100/provenance.md
docs/v100/patch-matrix.md
docs/v100/baseline.md
docs/v100/kv-path-trace.md
docs/v100/kv-format.md
docs/v100/kv-error-study.md
docs/v100/attention-selection.md
docs/v100/quant-weight-report.md
docs/v100/int8-kv-report.md
docs/v100/qualification.md
docs/v100/runtime-defaults.md
docs/v100/approvals.md
docs/v100/noise-floor.md
tools/v100/audit_turbomind_deltas.py
benchmarks/v100/attention/
benchmarks/v100/service/
benchmarks/v100/results/
tests/v100/
docker/v100/
```

Use machine-readable JSON or JSONL for raw benchmark results.

Map every final claim to a tracked report and a raw evidence file.

Record every donor file, symbol, revision, license, blob hash, and local replacement.

Preserve all OpenMMLab, NVIDIA, TileLang, Marlin, and donor notices.

Fail closed on unknown or incompatible provenance.

## Hard engineering rules

- Keep current LMDeploy model definitions as the source of truth.
- Keep current LMDeploy serving APIs as the source of truth.
- Preserve the generic upstream fallback.
- Add one patch family per commit.
- Give each patch family a stable ID.
- Add an opt-out for every new default-on path.
- Keep every unqualified path default-off.
- Use exact shape, data-type, layout, and architecture gates.
- Retain graph-owned buffers for the complete graph lifetime.
- Refresh graph metadata before every replay.
- Do not use descriptor-string matches when structured dimensions exist.
- Do not access private runtime fields across subsystem boundaries.
- Do not claim speed from unmatched workloads.
- Do not promote results without active-request and waiting-request telemetry.
- Run correctness, quality, and retrieval gates before default promotion.
- Do not hide rejected runtime code behind an environment variable.

Do not allocate a new full FP16 KV mirror.

One existing FP16 buffer is exempt from that rule.

TurboMind already flattens paged KV into a dense per-batch prefill buffer.

See `tmp_kv` in `src/turbomind/models/llama/unified_attention_layer.cc`.

That buffer is pre-existing upstream behavior, not a new mirror.

Phase 3 must bound it.

Phase 5 must not treat it as a rule violation.

## Phase timeboxes

Each phase has a timebox and an abort rule.

Stop and report when a phase exceeds its timebox.

Do not extend a timebox without operator approval.

| Phase | Timebox | Abort rule |
| --- | --- | --- |
| 0A source and capability preflight | 1 day | A required source does not resolve or a license fails. |
| 0B free-island baseline | 3 days | The small model fails TP1, or Qwen3.8 fails TP4 startup. |
| 0C TP8 baseline | 1 day | The operator withholds the second island. |
| 1 donor inventory | 3 days | The audit cannot separate machine facts from decisions. |
| 2 correction isolation | 5 days | The SIMT fault does not reproduce on a kernel fixture. |
| 3 quantized-weight port | 5 days | No family passes an independent qualification gate. |
| 4 attention shootout | 7 days | No candidate can host the block-scaled reader. |
| 5 block-scaled INT8 KV | 10 days | The error study shows no quality gain over the existing INT8 format. |

Phase 5 carries the campaign outcome, so it holds the largest budget.

A Phase 4 result that blocks Phase 5 is a campaign failure, not a Phase 4 success.

Record the actual elapsed time for each phase in the qualification report.

## Phase 0: lock sources and establish the baseline

### Phase 0A: source and capability preflight

Run these repository steps in order:

1. Fetch the `upstream` remote in `~/repos/cust-lmdeploy/lmdeploy/`.
2. Verify that tag `v0.16.0` resolves to `1208bf006bbac69f1f012ceafeeeb70f623b632c`.
3. Verify that every required checkout is clean.
4. Create a read-only checkout of `zh-nj/lmdeploy-v100` at `~/repos/cust-lmdeploy/lmdeploy-v100/`.
5. Fetch every donor remote and record the new head for each.
6. Create `docs/v100/source-lock.json`.
7. Create `experiment/lmdeploy-v100-next` from the product base.

Do not start the branch from the current local head.

The current local head is behind the product base.

Verify these upstream capabilities before donor work:

- Qwen3.8 checkpoint recognition.
- Qwen3.5 architecture mapping.
- Gated DeltaNet state allocation.
- Full-attention ownership.
- FP8 weight loading on SM70.
- CUDA graph support.
- OpenAI-compatible serving.
- TP1 startup with the small model.
- TP4 startup with Qwen3.8.

Fail closed on missing tensors, scales, or unsupported modules.

Record loaded tensor counts and expected tensor counts.

Record ignored-module decisions and quantization block shapes.

### Phase 0B: free-island baseline

Build current LMDeploy on the free V100 island.

Start with the selected small FP16 model.

Run TP1 with the small model first.

Run TP4 with the small model on the free NVLink island second.

Run Qwen3.8 at TP4 without donor kernels after the small model passes.

Do not run Qwen3.8 at TP1.

Record the KV budget table described in the qualification matrix.

Record these facts:

- Image digest.
- Source-lock SHA-256.
- Git revision.
- CUDA compiler and runtime versions.
- PyTorch version and CUDA build.
- NCCL version.
- GPU UUIDs and topology.
- CPU and NUMA affinity.
- Kernel registry contents.
- Loaded tensor and scale counts.
- KV capacity.
- Graph memory.
- Free memory after startup.
- Direct service launch command.

Stop if TP1 fails the declared reference contract.

#### Phase 0B result: PASS

Qwen3.8-27B-FP8 runs coherently at TP4 on the free island.

The FP8 checkpoint did not dispatch at all before this phase. It declares
`weight_block_size [128, 128]`, so `MakeQuantDesc` classified every weight as
`QuantType::kB`, and `kB` is declared only in `kernel_impl_sm90.h`. No SM70
kernel could match. Two commits fixed it, following `1Cat-vLLM` and
`marlin_v100` rather than writing a new mainloop:

- `c46dbf06` expands the block scale along N at load time, turning the
  `{128, 128}` block scale into the K-grouped `{128, 1}` form that the existing
  SM70 `Config_E4M3` tiles already implement. The expansion is exact.
- `f7b18471` casts the expanded scales to the compute dtype. Without it the
  model dispatched but emitted only `!`, because the checkpoint ships
  `weight_scale_inv` as BF16 while this run falls back to FP16 compute. Both
  are 16 bits wide, so no size check failed and the bytes were reinterpreted.

Measured facts:

| Fact | Value |
|---|---|
| Semantic gate | 8/8 |
| Long-context needle retrieval | PASS |
| Degenerate-output detector | False |
| Throughput | 208.1 tok/s, batch of 8, 512-token budget |
| Load time | 30.4 s |
| Memory | 23.35 GiB per GPU, 93.41 GiB total |
| Session length | 16384 |
| Compute dtype | FP16 fallback, `is_bf16_supported` is false |
| Island-2 GPU UUIDs | `c1aa8bd9`, `c275f176`, `d5302639`, `dd6f7287` |

Weights stay FP8. There is no dequantization to FP16 in memory.

#### Island isolation is mandatory

GPU indices are not a safe island boundary. `nvidia.com/gpu` requests are
satisfied by count, so the device plugin can assign any free device, and both
islands live on one node.

Pin island 2 by UUID with `NVIDIA_VISIBLE_DEVICES` and request no
`nvidia.com/gpu`, so the plugin cannot reassign devices. Guard every job:
abort before any allocation if an island-1 UUID is visible, if the visible
count is not 4, or if the island is not idle.

Island 1 UUIDs, never to be touched: `3ceb3a71`, `aa23eb12`, `f364b813`,
`07e14590`.

#### A test gate must assert meaning

The first passing run was wrong. The gate checked `if o.text.strip()` and
accepted 256 tokens of `!`. A gate must assert semantic content and detect
degenerate output.

Match after NFKC normalization. A correct `H₂O` answer failed an ASCII
substring test and looked like a numerical fault. Print the answer after
`</think>` rather than a blind tail, which hid a correct answer inside a
discursive reply.

Do not require PP2 x TP4 during the baseline.

Current TurboMind does not expose pipeline parallelism as a baseline configuration.

Treat pipeline parallelism as a separate future campaign unless the source lock proves native support.

### Phase 0C: TP8 baseline

Run TP8 only after operator approval.

Use both NVLink islands.

Record the exact rank-to-GPU map.

Restore the prior production service after the TP8 window.

TP8 is a correctness gate only.

TP8 spans two four-GPU islands, so the all-reduce crosses PCIe.

That cost does not represent any real deployment.

Do not derive a throughput claim from TP8.

Do not derive a latency claim from TP8.

Record TP8 memory and capacity facts, including the KV head doubling.

## Phase 1: automate the donor inventory

Write `tools/v100/audit_turbomind_deltas.py`.

The tool must inventory these sources:

```text
A: locked LMDeploy TurboMind
B: zh-nj/lmdeploy-v100, checked out in Phase 0A
C: 1Cat embedded TurboMind and its SM70 adapter
D: sglang-V100 TileLang attention
E: Tilelang-FA-V100, only if its license resolves
F: marlin_v100 SM70 quantized GEMM paths, only if its license resolves
```

Source C embeds a partial copy of an older LMDeploy tree.

That copy shares file paths with the product repository.

Compare by content hash, not by path, when classifying source C.

Separate machine facts from reviewed decisions.

Generate machine facts for files, commits, symbols, hashes, notices, and dependencies.

Store reviewed classifications and port decisions in `docs/v100/patch-matrix.md`.

Use these classifications:

```text
UPSTREAM_EQUIVALENT
UPSTREAM_REWRITE
V100_CORRECTNESS
V100_PERFORMANCE
NEW_QUANT_FORMAT
RUNTIME_ADAPTER_ONLY
EXPERIMENTAL_DEFAULT_OFF
OBSOLETE
UNKNOWN_PROVENANCE
INCOMPATIBLE_LICENSE
```

For each patch family, record:

- Patch-family ID.
- Commit SHA.
- Donor file and blob hash.
- Donor symbol.
- Current upstream equivalent.
- Dependencies.
- SPDX license identifier.
- Notice requirements.
- Compatibility decision.
- Test evidence.
- Service evidence.
- Port decision.
- Reviewer approval.

Fail the audit on unknown provenance, an unknown license, or an incompatible license.

Stop for operator approval after Phase 1.

## Phase 2: isolate `lmdeploy-v100` corrections

Do not copy its squashed root.

Audit the root against LMDeploy v0.12.1 by file and symbol.

Treat these post-root commits separately:

```text
c1e21e246108692d05408fcbedef2b48bb965a10
21dc748e9346e0dff18561722f73115b6a2ecf58
ead3054f10455ff8aebc1e73e2bb84b8bb047a1c
2c2b1ecfe1fcc5cef3c7a268c2668c3f19d29ce2
d7c29f88e44016d7a757850fe761c1b1b66181c8
```

### SM70 decode correctness

This section is a gate for the whole campaign.

Its result decides whether the campaign outcome is reachable.

Run it to completion before Phase 4 selects a kernel.

Do not start Phase 5 implementation before this gate closes.

Current LMDeploy uses MMA884 for SM70 prefill.

Current LMDeploy uses SIMT for SM70 decode.

The old fork reports incorrect TP8 output from that SIMT route.

Do not attribute the fault to SIMT before kernel isolation.

#### Why this gate decides the campaign

The Phase 1 audit established these facts from the donor checkout.

Donor commit `c1e21e24` states that a bisect blamed the SIMT decode route.

Its message reports garbage output for Qwen3.5-122B at TP8 with MoE.

Its remedy sets `SM70_PREFILL_USE_MMA_884` and `SM70_DECODE_USE_MMA_884` to 1.

The donor file `impl_884_decode.h` declares `using Tkv = T_`, so donor
`MMA_884_DEC` cannot consume a KV type distinct from the activation type.

#### Correction recorded at Phase 0A

An earlier reading of this campaign treated that donor constraint as a
campaign blocker. That reading was wrong. Record the correction here.

`MMA_884_DEC` is donor-only. It is not the blocker.

The product base contains no `MMA_884_DEC` symbol and no
`decoding_config.h`. The base replaced the donor compile-time dispatch with a
runtime registry. `decoding.cu` builds an `AttnDesc` and calls
`Registry::instance().Find(desc)`.

The base file `kernel/decoding_sm70_256.cu` already registers nine SM70
head_dim 256 decode kernels through `MMA_SIMT`:

```text
KT<half, half,    kH>   FP16 KV    kH in {1,2,3}
KT<half, uint8_t, kH>   INT8 KV    kH in {1,2,3}
KT<half, uint4_t, kH>   INT4 KV    kH in {1,2,3}
```

Qwen3.8 has 24 attention heads and 4 KV heads, so its query group size is 6
and the runtime selects `kH=3`. `KT<half, uint8_t, 3>` is therefore the exact
kernel that Qwen3.8 INT8 KV decode uses, and the base registers it today.

SM70 head_dim 256 quantized KV decode is an existing route. The campaign does
not need to invent it.

Registration is not proof of correct execution.

The gate below stays mandatory. It must confirm correct output and linked
SM70 machine code before Phase 5 builds on this route.

The donor evidence is not proof for the campaign target.

The donor reproducer is Qwen3.5-122B with MoE at TP8.

The campaign target is Qwen3.8-27B with Gated DeltaNet.

The donor remedy targeted a code structure that the base does not contain.

Treat the donor report as a risk to test, not as an established fault.

#### Gate outcomes

Record one of these outcomes in `docs/v100/kv-path-trace.md`.

Outcome 1: the SIMT decode route is correct for the campaign target.

The campaign proceeds as planned.

SIMT remains a valid host for the block-scaled reader.

Outcome 2: the fault reproduces, and it lies outside the SIMT decode kernel.

Fix the real cause, then re-run the gate.

Outcome 3: the fault reproduces inside the SIMT decode kernel.

Stop and report a campaign blocker before any Phase 5 implementation.

Do not implement a block-scaled format on a route with a known correctness fault.

Under outcome 3, evaluate these options and request an operator decision:

- Fix the SIMT decode fault directly, then continue.
- Port a `MMA_884_DEC` equivalent and extend it to accept a KV type distinct
  from `T`. The donor version cannot do this, so this option is new work.
- Restrict the campaign outcome to TP4, where the donor reports no fault.
- Restrict the campaign outcome to head dimensions that a ported 884 decode
  path supports.

Each option changes cost and scope, so none proceeds without approval.

Use this isolation order:

1. Compare SIMT with an eager FP32 or FP16 reference on identical tensors.
2. Compare eager execution with graph replay.
3. Compare TP1 with TP4 on the small model.
4. Compare TP4 with TP8 on Qwen3.8, after the operator reserves both islands.
5. Add MoE, GDN, long context, and service layers one at a time.

Steps 1 and 2 use kernel-level fixtures, not a loaded model.

Step 3 uses the small model, because Qwen3.8 does not fit one V100.

Use the fixed Qwen3.8 model record for the production reproducer.

Add a quantized-KV case to every isolation step.

Run each step with FP16 KV and with the existing INT8 KV format.

The block-scaled format does not exist yet, so the existing INT8 format is the
closest available proxy for the campaign outcome.

A fault that appears only with quantized KV is a direct campaign blocker.

If the isolated SIMT fault reproduces, do not port `MMA_884_DEC` by reflex.

A direct port cannot serve quantized KV, because it fixes `Tkv` to `T`.

Follow the gate outcomes above instead.

Do not port removed macro files.

The donor carries a `codegen/` directory that the product base does not have.

The Phase 1 audit lists those files as `donor_only`.

Add kernel, graph, TP8, and long-context regression tests.

### Weight-processing cache cleanup

Measure allocations before and after `_process_weights()`.

Synchronize each device before allocator cleanup.

Check prior CUDA launch errors before `torch.cuda.empty_cache()`.

Add per-device cleanup only when retained allocator memory exists.

Record startup cost and recovered KV capacity.

### Existing TurboMind INT8 KV trace

Treat the existing TurboMind INT8 KV path as the reference format.

Do not describe it as grouped or block-scaled INT8.

It uses per-token, per-KV-head asymmetric unsigned INT8 quantization.

For each token and KV head, it computes one `(scale, zero)` pair for the complete K vector.

It computes a separate pair for the complete V vector.

For Qwen3.8 full attention, one parameter pair covers all 256 dimensions.

Trace and document these current source boundaries in `docs/v100/kv-path-trace.md`:

```text
src/turbomind/kernels/attention/kv_cache_utils_v2.cu
    ProcessKV_v2
    invokeProcessKV_v2

src/turbomind/kernels/attention/quantization.h
    warp_stats
    ConvertKvCache<T, uint8_t>
    ConvertKvCache<uint8_t, T>

src/turbomind/kernels/attention/block.h
    block::Config
    block::Layout
    block::Head

src/turbomind/kernels/attention/block_iterator.h
    BlockIterator
    CacheIterFactory

src/turbomind/kernels/attention/attention_universal.h
    in-kernel KV write and quant-parameter storage
```

Trace the generic flatten, copy, and migration paths that consume the same physical block layout.

Record every size calculation and parameter offset that assumes two values per token and head.

Keep this format unchanged and available as the compatibility control.

### TurboMind MTP audit only

Audit the old `MTPPredictor` and rejection path in this phase.

Write the audit findings to `docs/v100/mtp-report.md`.

Do not implement MTP in this campaign.

See the separate MTP campaign section.

Do not restore the old global profiler.

Use Nsight Systems, CUPTI, or scoped runtime telemetry.

## Phase 3: port qualified SM70 quantized-weight work

Do not copy `awq_sm70_gemm.cu` as one unit.

Do not treat broad release commits as implementation units.

Create a commit-to-symbol map before each port.

Audit these 1Cat commits as reference ranges:

```text
a6a1b9abe18ecb3085058cb78d7c930d3ad128a7
e64d39aa7ee170a44dcdd3789e480d9aa24140ea
b78816657d9c25583d3f911443a7dfb6f1d463a7
15ace0f15986deadbdadce264dde1cfe7e305e2b
4f90bd6f5e6c80ec7858d557c8031da866a3052d
f228fa80d1e4cd9bd0036f7f47efc40719707488
4d0fcdf7ba2f602199f93cc8a85082712c092dd3
87ac589295ba64399695ef2237c37cffc0d8b71b
1d1daa789963af405005c4e209e2cfd4ad1b6af4
```

Prioritize these independently qualified families:

1. Dispatch-cache synchronization.
2. Active-expert scheduling.
3. Exact small-M tactic selection.
4. Bounded AWQ long-prefill workspace.
   Include the TurboMind `tmp_kv` prefill buffer in this family.
   That buffer scales with total prefill context.
   At long context it grows past one gibibyte.
   See `unified_attention_layer.cc`, near the `MAX_CTA_S` sizing term.
5. FP8 exact-dense long-prefill path.
6. Stable-pointer graph workspace ownership.
7. Deterministic compact reduction fixes.
8. Applicable Marlin SM70 grouped-scale mechanisms.

Current LMDeploy already contains SM70 AWQ, FP8 weight-only, and MXFP4 foundations.

These SM70 GEMM configurations exist today:

```text
src/turbomind/kernels/gemm/kernel/sm70_884_4.cu   Config_U4_d, Config_U4_g, Config_MXF4
src/turbomind/kernels/gemm/kernel/sm70_884_8.cu   Config_E4M3
src/turbomind/kernels/gemm/kernel/sm70_884_16.cu  Config_F16
```

Port only missing behavior with an approved patch-family record.

Write the Phase 3 findings and results to `docs/v100/quant-weight-report.md`.

Add fail-closed loader tests for these items:

- Tensor-name coverage.
- Scale-name coverage.
- Scale semantics.
- Ignored-module handling.
- Quantization block shape.
- Dense and MoE route ownership.
- Loaded tensor counts.

Keep these families default-off until independent qualification:

- NVFP4.
- Compact MXFP4 top-6 decode.
- DeepSeek exact M8 selectors.
- Broad FP8 router overrides.
- Low-level LM-head fused sampling.
- Dense FP16 donor paths.

Do not port the vLLM private custom-all-reduce coupling.

Do not port relative includes that escape the TurboMind subtree.

Design a native LMDeploy collective interface only after a separate measured case exists.

## Phase 4: select the SM70 attention kernel

Phase 4 selects the kernel that will host the block-scaled INT8 reader.

FP16 speed is the selection measure, not the selection purpose.

A fast FP16 kernel that cannot host the reader is not a valid winner.

See the hostability constraint below.

Keep Phase 4 benchmark-only until a candidate passes all gates.

Run the shootout on FP16 fixtures, because the new format does not exist yet.

Compare these candidates on identical FP16 fixtures:

```text
TurboMind SM70 MMA884 prefill
TurboMind SM70 SIMT decode
SGLang TileLang direct paged prefill
SGLang TileLang D256 dense-gather prefill
SGLang TileLang D256 split-KV prefill
SGLang TileLang grouped q=1 split-KV decode
Tilelang-FA-V100 applicable kernels
```

Do not include grouped INT8 in the first shootout.

The SGLang FP8 LUT route is not a grouped INT8 substitute.

### Selection constraint: block-scaled INT8 hostability

Hostability is the primary selection criterion.

Phase 5 needs a decode kernel that reads INT8 with per-block scales.

A Phase 4 winner that cannot host that reader blocks the campaign outcome.

Score every candidate on hostability before any speed comparison.

Record these facts for each candidate:

- Can the kernel load a separate scale array indexed by `d / group_size`?
- Can it dequantize in registers before the existing MMA or SIMT path?
- What is the estimated cost to add that reader?

Reject a candidate that cannot host the reader.

Rank the remaining candidates by FP16 speed.

Do not override a hostability rejection with an FP16 speed result.

If no candidate can host the reader, stop and report a campaign blocker.

The current SM70 decode kernel already hosts a `uint8_t` reader.

That kernel is a valid fallback winner only when the Phase 2 gate closes clean.

The current SM70 decode kernel is the SIMT route.

The Phase 2 gate tests that route for the donor-reported TP8 fault.

Do not select the SIMT route as the fallback winner under gate outcome 3.

Record the Phase 2 gate outcome in the selection report before selecting.

### TileLang ahead-of-time spike

TileLang compiles through Python and TVM at run time.

The decode hot path must not call Python.

Run a timeboxed spike before the shootout selects a TileLang candidate.

The spike must prove all of these items on SM70:

- Ahead-of-time cubin or fatbin export from TileLang.
- Load and launch from C++ without a Python interpreter.
- A stable C ABI for the exported entry point.
- SM70 machine code in the exported binary, not PTX only.

If the spike fails, drop every TileLang candidate from selection.

### Tiered shape matrix

Do not run the full Cartesian product.

Create a support manifest for every candidate.

Return `SKIP_UNSUPPORTED` with a reason for an invalid combination.

Use three tiers.

#### Tier 1: Qwen3.8 production shapes

Prioritize these shapes:

```text
head_dim: 256
query group: 6
batch: 1, 4, 8, 16
query length: 1, 2, 4, 8, 16, 32
context: 1K, 4K, 16K, 32K, 64K, 128K, near-256K
KV: FP16
page size: 16 and 64
```

Benchmark the grouped GQA D256 decoder first.

Benchmark the D256 split-KV prefill path second.

#### Tier 2: dispatch boundaries

Test supported boundaries around each dispatch predicate.

Cover head dimensions 64, 128, and 256 only for kernels that support them.

Cover query groups 1, 2, 3, 4, 6, and 8 only for valid head layouts.

Cover page boundaries, tail pages, empty prefixes, and maximum supported splits.

#### Tier 3: exploratory shapes

Run a bounded generic sweep after Tier 1 and Tier 2 pass.

Freeze seeds, strides, masks, page tables, sequence distributions, warmups, repetitions, and timer methods.

Record kernel duration, workspace, graph replay behavior, and numerical error for every case.

Use Nsight Compute for a small representative subset only.

Record HBM traffic and occupancy for that subset.

### Graph replay gates

Replay each captured shape with changed metadata.

Change these values across consecutive replays:

- Sequence lengths.
- Block tables.
- Query start locations.
- Request counts.
- Request slots.
- Active split counts.

Run the replay tests on every rank.

### Phase 4 exclusions

Context parallel is out of scope for this campaign.

The attention layer has a separate context-parallel workspace path.

Do not qualify, benchmark, or modify that path.

### Initial rejects

Do not port BFLA sparse attention.

Do not port DFlash verifier kernels during this phase.

Do not port the FP8 LUT bridge as an INT8 substitute.

Do not replace TurboMind generic prefill without a measured service win.

Prefer a TurboMind-native algorithm port over a Python runtime dependency.

If TileLang remains necessary, isolate it behind one stable C ABI.

Do not invoke Python from the C++ decode hot path.

## Phase 5: add grouped INT8 KV with block scaling

### Phase 5 goal

Phase 5 delivers the campaign outcome.

Every earlier phase is a prerequisite.

The goal is half the KV bytes of FP16, at FP16 quality.

Per token and KV head at head_dim 256, the costs are these:

```text
FP16 KV                              1024 bytes    baseline
existing per-token INT8               520 bytes    49% saving, quality control
block-64 symmetric INT8 candidate     528 bytes    48% saving, target format
```

Read those numbers against two different references.

Against FP16, the new format saves about 48 percent of KV bytes.

That saving is the campaign outcome.

Against the existing INT8 format, the new format costs about 1.5 percent more.

That 1.5 percent buys finer scale granularity.

The existing format computes one scale for all 256 dimensions.

The target format computes one scale for each 64-dimension block.

Phase 5 must prove that the finer granularity closes the quality gap to FP16.

### Phase 5 success and failure

Phase 5 succeeds when all of these hold:

- Block-scaled INT8 meets the FP16 quality gates in the service contract.
- Block-scaled INT8 beats the existing INT8 format on the error study.
- The format runs graph-safe on TP4.
- The measured KV capacity gain against FP16 matches the byte computation within tolerance.

Phase 5 fails when the error study shows no gain over the existing INT8 format.

On that failure, report the existing INT8 format as the retained INT8 path.

Do not ship a new format that only adds metadata cost.

Treat the existing TurboMind INT8 KV path as the compatibility control.

Keep its per-token, per-head asymmetric `uint8 + scale + zero` format unchanged.

Add block-scaled INT8 as a separate format with a separate dispatch identifier.

Do not reuse the existing INT8 KV format name or quantization policy value.

### Phase 5A: quantify the error trade

Capture representative pre-RoPE and post-RoPE K data plus V data from Qwen3.8.

Sample every full-attention layer across short, medium, and long contexts.

Keep K and V results separate.

Compare these candidates:

```text
CONTROL: per-head asymmetric uint8
    one scale and one zero per token, head, and K or V vector

A: block-128 asymmetric uint8
    two scale and zero pairs for D=256

B: block-64 asymmetric uint8
    four scale and zero pairs for D=256

C: block-128 symmetric int8
    two scales for D=256

D: block-64 symmetric int8
    four scales for D=256
```

Treat block-64 symmetric INT8 as the lead candidate, not a predetermined winner.

Write `docs/v100/kv-error-study.md` before the format decision.

Report these metrics by layer and context bucket:

- Maximum absolute error.
- Mean absolute error.
- Root mean square error.
- Relative error with a declared floor.
- Cosine similarity.
- SQNR.
- Saturation rate.
- Attention-score error.
- Attention-output error.
- Logit error.
- Argmax and top-k agreement.
- Retrieval and model-quality results.

Use the same captured tensors for every format.

Record outlier layers and tokens instead of reporting aggregate means only.

Select the format from error, metadata, memory, kernel, and service evidence.

### Phase 5B: freeze the new format

Freeze the complete format in `docs/v100/kv-format.md` before CUDA implementation.

For a symmetric winner, use this physical layout per layer:

```text
K signed INT8 payload
V signed INT8 payload
K FP16 block scales
V FP16 block scales
```

For an asymmetric winner, use this physical layout per layer:

```text
K unsigned INT8 payload
V unsigned INT8 payload
K FP16 block scales and zero values
V FP16 block scales and zero values
```

For the lead block-64 symmetric candidate with D=256, store four K scales and four V scales per token and head.

This gives 512 payload bytes and 16 scale bytes for K plus V.

The total is 528 bytes per token and head, before layout alignment.

Define and freeze these format values:

- The quantization group axis.
- The values per group.
- The payload signedness.
- The quantization formula.
- The scale data type.
- The scale and zero layout.
- The metadata alignment.
- The block and page interaction.
- The TP shard interaction.
- The rounding rule.
- The saturation range.
- The all-zero block rule.
- The tail-group rule.
- The K and V metadata offsets.
- The format version and migration behavior.

Do not change the selected format after qualification starts.

### Phase 5C: implement the writer and layout

Modify the traced TurboMind boundaries instead of adding an external cache converter.

#### New KV type tag

A block-scaled format cannot reuse `uint8_t` as the `Tkv` type.

The existing type carries a two-value metadata contract.

Add a distinct `Tkv` type tag for the new format.

That type flows through all of these sites:

```text
block.h            Config::t_bits, Config::q_bits
block.h            Layout::token_param_size
quantization.h     ConvertKvCache specializations
block_iterator.h   GetBlockIterFactory
kernel/decoding_sm70_*.cu   Registrar entries
kv_cache_utils_v2.cu        ProcessKV and FlattenKV dispatch
```

Decode SM70 at head_dim 256 registers nine kernels today.

That count is three `Tkv` types times three query-group tilings.

A new type adds three more kernels and more build time.

Record the kernel count and the build-time change before implementation.

#### Implementation order

Implement and verify this order:

1. CPU reference quantizer and dequantizer.
2. New `Tkv` type tag and its layout traits.
3. Block-layout size and offset calculations.
4. GPU writer in `ProcessKV_v2` or its narrow successor.
5. Generic reader through a new converter specialization.
6. Cache copy, flatten, migration, and slot-recycle paths.
7. Selected attention readers.
8. Graph-safe service integration.

For block-64, compute one parameter set for each contiguous 64-value region.

Do not reduce statistics across the complete 256-value head.

Keep the old `warp_stats` behavior for the compatibility format.

Add a block-group statistics helper for the new format.

Do not overload `ConvertKvCache<T, uint8_t>` with two incompatible metadata contracts.

Add explicit converter types or policies for the block-scaled formats.

#### The reader change is not converter-only

Record this constraint before any Phase 5C work.

The base reader loads one `(scale, zero)` pair for each token and each KV
head. `impl_simt.h` reads `param_K[n][0]` and `param_K[n][1]`, then applies
that single pair to every dimension fragment of the head.

At head_dim 256, block-64 needs four parameter groups for each token and
head.

A converter-only change would reuse the first group's scale for the other
three groups. That is silent numerical corruption, not a partial feature.

Phase 5C must therefore change the parameter fetch, not only the converter:

- Add a new `Tkv` type tag, so the two metadata contracts stay separate.
- Size and index the parameter storage for four groups for each head.
- Compute grouped writer statistics for each 64-value region.
- Load the group parameter that matches each dimension fragment.
- Update the cache copy, flatten, migration, and slot-recycle paths.
- Keep the existing per-head format working without change.

### Phase 5D: dequantize on the two read paths

SM70 has two distinct KV read paths.

Treat them separately.

#### Decode path: dequantize inside attention

The decode kernel reads the paged cache directly through the block iterator.

For a symmetric retained kernel, use this data path:

```text
signed INT8 HBM load
FP16 scale load indexed by d / group_size
INT8 to FP16 conversion in registers
FP16 multiply by scale
existing FP16 SIMT or split-K attention
```

For an asymmetric retained kernel, include the block zero value in the same direct dequantization path.

#### Prefill path: dequantize in the flatten kernel

The SM70 prefill kernel does not read the paged cache directly.

The attention layer runs `ProcessKV_v2`, then `FlattenKV_v2`, then the attention kernel.

The prefill kernel reads a dense FP16 buffer through the linear iterator.

That buffer is `tmp_kv`, sized per batch from the prefill key sum.

Add the block-scaled reader to the flatten kernel converter.

Do not add a block-scaled reader to the SM70 prefill attention kernel.

Record that the prefill path dequantizes to FP16 before attention.

#### Memory rules

Do not materialize a full FP16 KV cache in HBM.

Do not add a new full-cache scratch mirror.

The existing `tmp_kv` prefill buffer is exempt, because it is upstream behavior.

Phase 3 bounds that buffer.

Include metadata loads and address arithmetic in kernel benchmarks.

### Phase 5E: qualification

Verify writer and reader agreement independently.

Test these boundaries:

- Every 64-value and 128-value group boundary.
- Tail groups for non-D256 regression shapes.
- All-zero groups.
- Constant nonzero groups.
- Saturated positive and negative values.
- Mixed old-format and new-format requests only if the runtime supports both safely.
- Mixed cache blocks.
- Page boundaries.
- Shuffled block tables.
- Recycled request slots.
- K and V metadata offsets.
- TP shard boundaries.
- Changed-input graph replay.
- Cache copy and migration.

Run a separate block-scaled INT8 attention matrix after FP16 candidate selection.

Compare the retained candidate against both FP16 KV and existing TurboMind INT8 KV.

Do not compare block-scaled INT8 with the SGLang FP8 LUT path as equivalent formats.

Keep the new format default-off until its complete correctness and service gates pass.

### Phase 5F: promote the new format

Promotion to default-on is the campaign outcome.

Request operator approval before the promotion, as the execution contract requires.

Promote only when all of these hold:

- Every correctness gate in the service quality contract passes.
- The error study shows a quality gain over the existing INT8 format.
- Quality sits within the declared tolerance of the FP16 KV baseline.
- The measured KV capacity gain against FP16 matches the projection.
- No performance gate regresses outside its declared tolerance.
- TP4 graph replay passes on every rank.

Promote for the Qwen3.8 record only.

Keep FP16 KV as the default for every other model.

Provide an opt-out flag for the new default, as the hard rules require.

Keep the existing INT8 format reachable and unchanged.

Record the promotion decision and its evidence in `docs/v100/runtime-defaults.md`.

## Separate campaign: TurboMind MTP

MTP is out of scope for this campaign.

The remaining text in this section is a design input for that later campaign.

### Why MTP is a separate campaign

TurboMind contains no MTP implementation.

A search of `src/turbomind/` finds no MTP symbol.

MTP exists only in the PyTorch engine, under `lmdeploy/pytorch/spec_decode/`.

TurboMind has draft-token accounting hooks only.

Those hooks are the `alpha` and `beta` fields in `src/turbomind/engine/request.h`.

Their comments read `draft_len + input_len` and `draft_len + {0,1}`.

MTP in TurboMind is therefore new subsystem development, not a port.

The donor fork targets v0.12.1, which predates the current builder and weight refactor.

Its MTP code is not a usable port source.

### Entry conditions for the MTP campaign

Start the MTP campaign only after all of these conditions hold:

- This campaign closes with every retained path qualified.
- `docs/v100/mtp-report.md` records the Phase 2 audit findings.
- A separate MTP design document exists.
- The operator approves that design document.

### Design inputs carried forward

Keep these concepts independent:

```text
checkpoint MTP layer count
speculative depth
verifier query size = speculative depth + 1
accepted-prefix length
```

Qwen3.8 contains one checkpoint MTP layer.

Start with speculative depth one and a two-row verifier query.

Add depth three and a four-row verifier query only after depth one passes.

Do not call depth three a three-layer MTP model.

Use greedy verification for initial isolation.

Then run the checkpoint generation profile and stochastic rejection tests.

Add sequential-oracle tests for every accepted-prefix length.

Compare these recurrent-state facts after acceptance and rejection:

- State IDs.
- State tensors.
- Block ownership.
- Rejected-token recovery.
- Request-slot reuse.
- State-block rollover.
- Prefix-cache resume.

Require verifier-logit parity under the declared numeric contract.

Do not infer token counts from streaming response chunks.

Use returned completion-token usage for service throughput.

Use internal draft, accepted-token, and accepted-position counters for MTP analysis.

End of the MTP design inputs.

## Correctness contracts

Define each contract before its test runs.

### Kernel contract

Use a CPU FP32 or trusted eager reference.

Use bitwise comparison only when the same algorithm and operation order require it.

Record absolute and relative tolerances for every other case.

### Accumulator contract

Accumulate every GEMM reduction in FP32 by default.

Do not accumulate a full K-loop in FP16 on this model. The checkpoint is BF16.
BF16 carries 8 mantissa bits with the exponent range of FP32, and FP16 carries
11 mantissa bits with a maximum finite value of 65504. An FP16 accumulator
therefore removes exponent range that the model was trained to use.

Two failure modes apply, and only the first is loud:

1. Overflow. A partial sum that reaches 65504 becomes infinity and never
   recovers.
2. Swamping. Once the accumulator is large, small addends round away. With
   `eps = 2^-11`, an accumulator near 1024 discards every addend below 0.5.

Swamping is the dangerous case. It raises no error. It removes the small
contributions that a long reduction must sum, so the model gets quietly worse
instead of failing.

The reductions in this model are long enough for this to matter. At TP4,
`gate_up_proj` reduces over K=5120 and `down_proj` over K=4352, giving a
random-walk relative error near 0.035, which is far above a `1e-3` kernel
tolerance.

`ds4/kernels/tc-grid/kernels/mma_sm70.cuh` provides an FP16-accumulate
`m8n8k4` wrapper, and its source comments claim about double the tensor-core
peak. Treat that as a candidate, not a plan. In `v12_kernels.cuh` and
`v13_kernels.cuh`, `c_frag` is declared `half`, zeroed once, and accumulated
across the whole K-loop, which is the unsafe form.

If FP16 accumulation is attempted, it must satisfy all of the following.

- Keep each FP16 chain short and reduce across chains in FP32. The SplitK
  kernel already does this, because `KS > 1` reduces partials with an FP32
  `atomicAdd`. At `KS = 8`, the per-chain length for `gate_up_proj` falls from
  5120 to 640.
- Put the route behind a named switch that defaults to off.
- Qualify it on end-to-end output quality, not on a kernel tolerance band. A
  per-element bound on one GEMM says nothing about degradation accumulated
  across 48 layers or about long-context behavior.
- Compare greedy-decode outputs against the FP32-accumulator build on frozen
  prompts at short and long context. Treat any systematic divergence as a
  failure.

A kernel tolerance band is never sufficient evidence for this change.

### Logit contract

Record maximum absolute error, maximum relative error, top-k overlap, and argmax agreement.

Do not require cross-topology bitwise equality from FP16 reductions.

### Token contract

Use frozen prompts, tokenizer files, chat templates, generation parameters, and seeds.

Use deterministic token parity only under a deterministic execution profile.

### Distribution contract

Use declared sample counts and acceptance thresholds for stochastic paths.

Record the complete generation configuration.

### Service quality contract

Require all applicable gates:

- TP1 reference parity with the small model.
- TP4 parity with Qwen3.8.
- TP8 parity after operator approval, as a correctness gate only.
- Changed-input graph replay.
- Shuffled page-table parity.
- Mixed sequence-length parity.
- Recycled request-slot parity.
- 8K retrieval.
- 32K retrieval.
- 128K retrieval.
- Near-256K retrieval.
- Repeated-prefix retrieval.
- Fixed-corpus output hashes under a deterministic profile.
- Sampling distribution tests.
- Perplexity comparison.
- Qwen3.8 model-specific evaluation.
- Tool-call evaluation.
- Long-output state-stability evaluation.

Assert on response content, not HTTP status or token count alone.

Fail on repeated unknown tokens, empty content, non-finite logits, or missing scales.

## Performance gates

### Noise floor

Measure the noise floor before any candidate comparison.

Run a control against an identical control on the same node.

Use the same trace, the same warmup, and the same window length.

Run that pair across at least three clean service starts.

Record the observed spread in `docs/v100/noise-floor.md`.

Record the confidence-interval method and the sample count.

Set every promotion threshold above the measured floor.

The numeric thresholds in this section are provisional.

Replace them with measured values before the first promotion decision.

Do not promote on a margin at or below the noise floor.

TP8 runs produce no performance claim, so they need no noise floor.

### Measurement method

Use direct service endpoints.

Do not benchmark through a response cache.

Vary prompts when any cache can exist.

Warm all JIT and graph paths before measurement.

Use fixed request traces for matched service comparisons.

Use two-to-four-minute post-warmup windows.

Run at least eight warmed samples for each retained comparison.

Interleave control and candidate runs.

Repeat the final comparison across at least three clean service starts.

Record the median, spread, and confidence interval.

Record these metrics:

- Aggregate tokens per second.
- Tokens per second per active slot.
- TTFT.
- Inter-token latency.
- Active requests.
- Waiting requests.
- KV occupancy.
- All-rank GPU utilization.
- All-rank power.
- All-rank memory.
- Kernel duration.
- Graph memory.
- Free-memory minimum.
- Thermal and clock state.

Promote a kernel only when it passes correctness and one performance rule:

```text
>= max(1.5%, measured noise floor + margin) matched service throughput gain,
   with the confidence interval above zero
or
>= 0.4 ms critical-step reduction with no matched service regression
or
>= 5% isolated kernel gain with no matched service regression
```

Derive the margin from the noise-floor study.

State the chosen margin and its basis in the qualification report.

Require no TTFT or latency regression outside the declared tolerance.

Reject a candidate when it adds a full-cache FP16 mirror.

Reject a candidate when it doubles launch count without a service win.

Reject a candidate when one rank diverges numerically.

Reject a candidate when the result disappears after a clean restart.

## Homelab build and rollout

Use a dedicated development pod on `gpu-01` for kernel work.

Request at most four GPUs before the TP8 approval gate.

Verify that Kubernetes assigned the free island before each run.

Use the homelab registry for integrated images:

```text
localhost:32000/lmdeploy-v100-next:<tag>
```

Use image tags in this form:

```text
vN-v100-lmdeploy-next
```

Create these homelab artifacts:

```text
manifests/lmdeploy-v100-next/00-build.yaml
manifests/lmdeploy-v100-next/10-deployment.yaml
manifests/lmdeploy-v100-next/20-service.yaml
manifests/lmdeploy-v100-next/README.md
manifests/lmdeploy-v100-next/bench-history.txt
```

Roll an isolated candidate before any production route change.

Keep the previous image and manifest as rollback artifacts.

Do not promote an ad hoc `kubectl set image` state.

## Commit policy

Use concise, imperative commit subjects.

Keep each commit independently buildable and testable.

Add tests in the same commit as each behavior change.

Reference the patch-family ID in the commit body.

Add provenance trailers for donor-derived changes.

Record rejected experiments in `docs/v100/qualification.md`.

Keep the final worktree clean.

## Final deliverable

### Primary outcome

Return the block-scaled INT8 KV result first:

1. The frozen format specification.
2. The error study against FP16 KV and against the existing INT8 format.
3. The correctness and service gate results.
4. The measured KV capacity gain against FP16.
5. The promotion decision and its evidence.
6. The opt-out flag and its default.

State plainly whether the campaign met its outcome.

If it did not, state which gate failed and what the retained INT8 path is.

### Supporting evidence

Return these items:

1. The source lock and its SHA-256 digest.
2. The exact branch and commit list.
3. The provenance matrix.
4. The patch classification matrix.
5. The baseline and topology report, including the KV budget table.
6. The attention selection report, with hostability scores and FP16 raw data.
7. The TP1 small-model, TP4 Qwen3.8, and approved TP8 correctness results.
8. The SM70 quantized-weight report.
9. The measured noise floor and the derived promotion margin.
10. The retained runtime defaults.
11. Every default-off experimental path.
12. Every rejected candidate and reason.
13. The reproducible container recipe, lock files, digest, and launch command.
14. The homelab manifests and rollback procedure.
15. The approval record.
16. The MTP audit report, as an input to the separate MTP campaign.

Do not require PP2 x TP4 for this campaign.

Do not include MTP in this campaign.

Do not call the campaign complete until every retained path passes its declared gate.
