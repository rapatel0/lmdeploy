# Provenance report

This report records machine facts from Phase 0A and Phase 1.

It records no port decision. Port decisions belong in `patch-matrix.md`.

## Product base

| Item | Value |
| --- | --- |
| Repository | InternLM/lmdeploy |
| Ref | `v0.16.0` |
| Commit | `1208bf006bbac69f1f012ceafeeeb70f623b632c` |
| Branch | `experiment/lmdeploy-v100-next` |
| License | Apache-2.0, LICENSE file at root |

The local checkout was 73 commits behind this tag before Phase 0A.

Phase 0A fetched the `upstream` remote and branched from the tag.

The `origin` remote is a fork and does not carry the tag by itself.

## Source lock

The lock file is `docs/v100/source-lock.json`.

Regenerate it with `tools/v100/make_source_lock.py`.

The tool fails closed on a missing checkout, a moved head, a dirty donor
worktree, a wrong campaign branch, a detached head, a missing license file,
or an unresolved license.

The product base is exempt from the dirty check only, because campaign edits
live in that repository. Its content still enters the audit at the locked
commit, so neither working-tree edits nor campaign commits can change any
recorded digest.

The lock is immutable. The tool refuses to overwrite an existing lock, and a
replacement needs the explicit `--relock` flag at an approved phase boundary.
A failed run writes `source-lock.failed.json` and leaves the lock untouched.

## Licenses

| ID | Source | SPDX | Evidence |
| --- | --- | --- | --- |
| A | lmdeploy | Apache-2.0 | LICENSE file at root |
| B | zh-nj/lmdeploy-v100 | Apache-2.0 | LICENSE file at root |
| C | 1CatAI/1Cat-vLLM | Apache-2.0 | LICENSE file at root |
| D | haohervchb/sglang-V100 | Apache-2.0 | LICENSE file at root |
| E | haohervchb/Tilelang-FA-V100 | MIT | README.md License section. No LICENSE file at root. |
| F | zhinianqin/marlin_v100 | Apache-2.0 | Per-file headers in `csrc/`. No LICENSE file at root. |

Sources E and F carry no LICENSE file at their roots.

The master specification requires a resolved license before Phase 1.

Both resolved, so both stayed in the campaign.

Source F carries the Marlin and Neural Magic Apache-2.0 notice in
`csrc/quantization/marlin/marlin.cu`. Preserve that notice on any port.

## Donor inventory

The inventory file is `benchmarks/v100/results/donor-inventory.jsonl`.

The summary file is `docs/v100/donor-inventory-summary.json`.

Regenerate both with `tools/v100/audit_turbomind_deltas.py`.

The audit reads every file at its locked commit, never from the working tree
and never from the branch head. Each record carries
`content_source: locked_git_commit`.

The product base is indexed at its locked commit, so campaign commits cannot
move the comparison base. The summary records the locked commit, the current
head, and the current branch as separate fields.

The audit revalidates every checkout against the lock before any read. It
fails on commit drift, tree drift, dirty donor state, or license digest drift.
The product base uses the ancestor rule; donors must match exactly.

Each record also carries `symbols`, `dependencies`, and `revisions` facts,
which give the patch matrix a starting commit-to-symbol map.

The symbol scan is lexical, not a parse. Every record states its method in
`symbol_extraction` and its cap state in `symbols_truncated`. The dependency
and revision facts carry the same truncation flags.

A reviewer must confirm every symbol before a port.

An empty symbol list is not proof that a file defines no symbol.

Symbol coverage on the current run is 615 of 641 records. The six source
files with no symbol are wrapper headers, package initialisers, and a
binding file that genuinely declare none.

### Delta states

These states are content facts, not campaign classifications:

| State | Meaning |
| --- | --- |
| `identical_content` | Byte-identical to some product-base file |
| `modified` | A product-base file shares the path, content differs |
| `donor_only` | No product-base counterpart |

Comparison uses content digests, not paths. Donor C embeds a partial copy of
an older LMDeploy tree at shared product paths, so a path comparison alone
would misreport it.

### Counts

Total records: 641. Product base indexed 1139 source digests and 22 build
digests across 1187 tracked paths.

Source and build files are compared against separate digest indexes, so a
build file is never matched against a source file.

| ID | Source | Files | Source deltas | Build deltas |
| --- | --- | --- | --- | --- |
| B | zh-nj/lmdeploy-v100 | 445 | 142 identical, 181 modified, 104 donor-only | 16 modified, 2 donor-only |
| C | 1CatAI/1Cat-vLLM | 139 | 76 identical, 57 modified, 6 donor-only | none |
| D | haohervchb/sglang-V100 | 10 | 10 donor-only | none |
| E | haohervchb/Tilelang-FA-V100 | 9 | 9 donor-only | none |
| F | zhinianqin/marlin_v100 | 38 | 36 donor-only | none |

### Reading the counts

The delta columns count compared files only.

A compared file is a source file or a build file. Every other tracked file
reads as `not_compared`.

Source F also tracks two `.gitignore` files, which fall in the `other`
category and carry no delta.

Source B shares 142 files with the product base without change. Those files
need no port. Its 181 modified files are the audit surface for Phase 2.

Source C embeds 139 files, of which 76 are byte-identical to upstream. Its
57 modified files differ from a product-base file at the same path, and its
6 unmatched files have no product-base counterpart.

Those states record content facts only. They do not classify behavior.
Behavior classification belongs in `patch-matrix.md`.

Sources D, E, and F share no path with the product base, so every compared
file reads as `donor_only`. That state means "no counterpart", not "new work".

## Phase 1 finding: SM70 decode routing

The audit surfaced a campaign-level risk. Full detail sits in the master
specification, under the Phase 2 SM70 decode correctness gate.

Summary of the machine facts:

1. Donor commit `c1e21e24` reports a bisect blaming the SIMT decode route for
   garbage output on Qwen3.5-122B at TP8 with MoE. Its remedy enables
   `MMA_884_DEC`.
2. Donor `decoding_config.h` routes head_dim 256 with quantized KV to
   `MMA_SIMT` unconditionally.
3. Donor `impl_884_decode.h` declares `using Tkv = T_`, so `MMA_884_DEC`
   cannot accept a KV type distinct from the activation type.
4. The product base contains no `MMA_884_DEC` symbol.

Consequence: any quantized KV at head_dim 256 must decode on the SIMT route,
and the donor's own remedy cannot serve it.

The donor reproducer is Qwen3.5-122B with MoE. The campaign target is
Qwen3.8-27B with Gated DeltaNet. The report is therefore a risk to test, not
an established fault for this target.

Phase 2 must close this gate before Phase 4 selects a kernel.

## Reproduction

```bash
cd ~/repos/cust-lmdeploy/lmdeploy
# The lock already exists and is immutable. Add --relock only at an approved
# phase boundary.
python3 tools/v100/make_source_lock.py --out docs/v100/source-lock.json
python3 tools/v100/audit_turbomind_deltas.py \
    --lock docs/v100/source-lock.json \
    --out benchmarks/v100/results/donor-inventory.jsonl \
    --summary docs/v100/donor-inventory-summary.json
```

Both commands exit non-zero on any provenance failure.

The summary binds both artifacts. It records the SHA-256 digest of the source
lock and the SHA-256 digest of the inventory file.

| Artifact | SHA-256 |
| --- | --- |
| `docs/v100/source-lock.json` | `0cc1230a863b60f9dea4ebb64ae2febdc620ee223d9efb1f3ca3b9a56cbe4dfc` |
| `benchmarks/v100/results/donor-inventory.jsonl` | `bb7b0e11b473bd6400062dbe4754abc54cb19be8202b71fa6a8480e6829126ca` |

Both tools hash every license file at its locked commit, not from disk, so a
dirty product-base worktree cannot change a recorded license digest.

## Base pinning

The product base entry records `resolved_commit` as the expected tag commit,
never the current head. The current head is recorded separately as
`observed_head`.

That rule keeps a permitted re-lock pinned to `v0.16.0`. Without it, a
re-lock after a campaign commit would advance the comparison base, because
the ancestor rule accepts any descendant.

## Phase 0B build, SM70 wheel

Built on gpu-01 in-cluster. Compilation needs nvcc, not a GPU, so the build
Job requests no GPU and leaves both GPU islands untouched.

| Item | Value |
| --- | --- |
| Source commit | `43908d2f` |
| Source archive sha256 | `1d872876ccd903ae...` (git archive of HEAD) |
| Toolchain image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| nvcc | 12.8.93 |
| Python | 3.12 |
| Architecture | `70-real`, pinned through `CUDAARCHS` |
| Wheel | `lmdeploy-0.16.0-cp312-cp312-linux_x86_64.whl`, 28 MB |
| Wheel sha256 | `a8f975227b5f2157ebc63b7969c21b52fc834618edc5211027f33e195be2d6f7` |
| Build duration | 9m20s |

### Gates

Both gates passed on the retained wheel.

```
architectures compiled: 70 (count 1)
PASS: single architecture, compute_70

=== scanning 2 libraries for sm_70 machine code ===
  OK   _turbomind...so: 91 sm_70 ELF sections
PASS: 1 libraries contain sm_70 machine code
```

### Three defects found while building

The first two builds "succeeded" in the sense that they produced an
importable wheel, yet both were wrong. Recording them, because each one fails
silently.

**1. The architecture pin was inert.** Exporting `CMAKE_CUDA_ARCHITECTURES`
does not reach CMake. `python -m build` isolates the environment, and
`setup.py` passes a fixed `cmake_configure_options` list that omits the
architecture. The first wheel was 331 MB and carried sm_70, sm_75, sm_80,
sm_86, sm_89, sm_90a, sm_100a, and sm_120a. It would have run correctly on a
V100 while hiding that the pin did nothing.

`CMAKE_ARGS` is not a fix either. `cmake_build_extension` 0.6.1 builds its
argument list from `cmake_configure_options` plus its own `-D` options and
never reads `CMAKE_ARGS`.

`CUDAARCHS` is the correct channel, because CMake reads it natively when it
enables the CUDA language, and `CMakeLists.txt` guards its default list with
`if (NOT CMAKE_CUDA_ARCHITECTURES)`. The wheel dropped to 28 MB.

**2. The build was not idempotent.** A retry inherited a root-owned `build/`
tree on the PVC whose `CMakeCache.txt` pinned `Python3_ROOT_DIR` to a deleted
temporary directory. The retry therefore failed for a different reason than
the original pod, which obscured the real error. The script now removes
`build/` and `lmdeploy.egg-info` first, and the Job uses `backoffLimit: 0` so
a failure surfaces honestly instead of being masked by a retry.

**3. The pin verifier itself was wrong.** It used
`find -name build.ninja | head -1`, which returns an arbitrary dependency
subbuild such as `concurrentqueue-subbuild`. Those files contain no CUDA
flags, so the check found zero architectures and failed a correct build. It
now scans every ninja file under the turbomind tree and aggregates, which
also catches a dependency compiling for an unwanted architecture.

The original verifier confirmed that sm_70 was *present* but never that other
architectures were *absent*. That asymmetry is what let defect 1 through.

## Phase 0B smoke test, verified on a V100

Ran on gpu-01 against a Tesla V100-SXM2-32GB, compute capability 7.0.

```
torch 2.9.1+cu128
arch_list ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
PASS: torch ships sm_70 and the device is (7, 0)
PASS: fp16 matmul executed, result finite
PASS: _turbomind imported          (61 exported names)
PASS: TurboMind importable
ALL SMOKE CHECKS PASSED
```

### The runtime torch pin

Modern torch dropped V100. `2.12.1+cu130` is built for compute capability 7.5
and above, so on a V100 it installs cleanly, reports capability `(7, 0)`
correctly, and then fails at the first kernel launch. Measured
`torch.cuda.get_arch_list()` on this node:

| Build | sm_70 |
| --- | --- |
| `2.12.1+cu130` | absent |
| `2.9.1+cu126` | present |
| `2.8.0+cu126` | present |
| `2.9.1+cu128` | present |

Pinned `torch==2.9.1` and `torchvision==0.24.1` from the cu128 index, in
`tools/v100/constraints-v100.txt`. It is the newest verified build carrying
sm_70, and its CUDA minor version matches the 12.8.1 build toolchain, so the
build and runtime cannot drift.

Install with `--index-url`, never `--extra-index-url`. An extra index does not
constrain resolution, it only adds candidates, which is how pip selected
`2.12.1+cu130` while cu128 was supplied as an extra.

### Two further defects, both of which reported success

**The harness masked a failure.** The first smoke Job reported SUCCEEDED while
its log showed `smoke_rc=1`. The final command was a pipeline, so the exit
status of the failing step was discarded. `smoke_v100.sh` now runs under
`set -euo pipefail`.

**`.gitignore` swallowed a build input.** A blanket `*.txt` rule at line 84
excluded `tools/v100/constraints-v100.txt`. The commit succeeded, the working
tree looked clean, and the file existed on disk, so nothing looked wrong. On a
fresh clone the constraint would have been missing, pip would have resolved
cu130 again, and the failure would have appeared as an unexplained kernel
launch error. Force-added.

### Note on the import spelling

`_turbomind` is a top-level module located through a `sys.path` insert into
`lmdeploy/lib`, performed by `lmdeploy/turbomind/turbomind.py`. It is not a
submodule of `lmdeploy.turbomind`, so `from lmdeploy.turbomind import
_turbomind` always fails regardless of build health. Any check must mirror the
real import path.

## Phase 0B TP1 baseline, Qwen3.5-4B

Passed on the free island, GPU 4, with a clean exit.

| Fact | Value |
| --- | --- |
| Model | `/srv/models/Qwen3.5-4B`, `Qwen3_5ForConditionalGeneration` |
| GPU UUID | `GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b` |
| Device | Tesla V100-SXM2-32GB, compute capability 7.0 |
| torch | 2.9.1+cu128 |
| NCCL | 2.27.5 |
| nvcc | 12.8.93 |
| Resolved dtype | float16 |
| session_len | 8192 |
| cache_max_entry_count | 0.6 |
| Load | 18.7 s |
| Used after startup | 24.13 GiB |
| Free after startup | 7.60 GiB |
| Decode | 96 tokens in 0.88 s, 109.5 tok/s |
| Exit | 0, `CLEAN_SHUTDOWN_OK` |

The dtype line confirms the capability preflight:

```
converter.py:148 - data type fallback to float16 since
torch.cuda.is_bf16_supported is False
```

SM70 has no BF16, so LMDeploy downgrades rather than failing. A BF16
checkpoint therefore buys nothing on this hardware.

### Two log lines that look like failures and are not

**`Warm-up for 8320 tokens failed with status 6`.** Status 6 is `kTooLong` in
`src/turbomind/engine/request.h:126`, meaning history plus prompt exceeds
`session_len`. TurboMind probes warm-up sizes above the configured limit, so
the rejection is correct behavior. The first run configured `session_len`
4096 and produced three such lines at 6144, 8192, and 8320. Raising
`session_len` to 8192 left only the 8320 probe.

**`terminate called without an active exception`, exit 134.** This was a
harness defect, not an engine defect. `TurboMind.close()` and
`AsyncEngine.close()` exist, and the first run called neither, so the engine
internal thread was still running at interpreter shutdown. Calling
`pipe.close()` before exit produced `CLEAN_SHUTDOWN_OK` and exit 0.

Any long-running service or benchmark harness in later phases must close the
engine explicitly. An abort at teardown would otherwise corrupt the exit
status of an otherwise valid benchmark.

### A harness defect worth recording

The first TP1 Job reported SUCCEEDED while the container exited 134. The
script ended on a heredoc whose status was never propagated, so Kubernetes saw
the shell's status rather than Python's. The job script now ends with
`exit $RC`.

## Phase 0B TP4 baseline, Qwen3.5-4B

Passed on the free island with a clean exit.

| Fact | Value |
| --- | --- |
| Topology | NV1/NV2 mesh, every pair NVLink, no SYS hop |
| NUMA | node 1, CPUs 20-39 and 60-79 |
| NCCL | 2.27.5 |
| Load | 17.1 s |
| Per-GPU used | 20.61 GiB, identical on all four ranks |
| Per-GPU free | 11.12 GiB |
| Decode | 96 tokens in 0.46 s, 209.7 tok/s |
| Exit | 0, `CLEAN_SHUTDOWN_OK` |

TP4 decode is 1.92x TP1 decode, 209.7 against 109.5 tok/s, which is
consistent with NVLink collectives rather than a PCIe fallback.

Memory is identical to two decimal places across all four ranks, so the shard
split is even and no rank carries a padded remainder.

### A note on GPU indices

The container sees GPUs 0 through 3. Those are physical GPUs 4 through 7,
remapped by the device plugin. `nvidia-smi topo -m` inside the container
reports NUMA node 1 and CPU affinity 20-39 and 60-79, which identifies the
second island. Any rank-to-GPU map recorded from inside a container must be
read with that remapping in mind.
