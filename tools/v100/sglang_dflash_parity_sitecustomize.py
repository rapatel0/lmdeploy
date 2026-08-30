"""Trace-only SGLang DFlash hooks loaded through sitecustomize.

This file monkey-patches the installed reference runtime in memory; it never
changes or ships SGLang source. Set SGLANG_DFLASH_PARITY_DIR to enable it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_TRACE_ROOT = os.environ.get("SGLANG_DFLASH_PARITY_DIR", "")
if _TRACE_ROOT:
    import sglang.srt.models.dflash as _df
    import torch

    _seen: set[str] = set()
    _graph_refs: dict[str, torch.Tensor] = {}
    _target_trace: dict[str, torch.Tensor] = {}
    _target_trace_dtypes: dict[str, int] = {}
    _graph_flushed = False
    _ordinal = 0
    _arm_file = os.environ.get("SGLANG_DFLASH_PARITY_ARM_FILE", "")
    try:
        _target_position = int(os.environ.get("SGLANG_DFLASH_TARGET_POSITION", "999"))
    except (TypeError, ValueError):
        _target_position = 999
    _target_row = -1

    def _armed() -> bool:
        # SGLang executes synthetic DFlash blocks while warming kernels during
        # server startup. The external client creates this shared marker only
        # after health_generate is ready, immediately before the audited real
        # request. Keeping every hook inert until then prevents a valid-looking
        # trace of dummy token IDs and avoids synchronizing warm-up kernels.
        return not _arm_file or Path(_arm_file).exists()

    def _safe_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _tp_rank() -> int:
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            return int(get_tensor_model_parallel_rank())
        except Exception:
            return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    def _tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            return next((item for item in value if isinstance(item, torch.Tensor)), None)
        return None

    def _dump(name: str, value, *, last_row: bool = False) -> None:
        global _ordinal
        if name in _seen:
            return
        tensor = _tensor(value)
        if tensor is None or tensor.numel() == 0:
            return
        if last_row and tensor.ndim:
            tensor = tensor.reshape(-1, tensor.shape[-1])[-1:]
        # Never serialize while CUDA is capturing, even when the external arm
        # marker already exists: DFlash may lazily recapture on the first real
        # request. Keep the latest graph-owned address and flush it only after
        # replay. This prevents armed lazy-capture placeholders from winning.
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = False
        if capturing:
            _graph_refs[name] = tensor.detach()
            return
        if not _armed():
            return
        tensor = tensor.detach().contiguous().cpu()
        dtype_name = {
            torch.float16: "f16",
            torch.float32: "f32",
            torch.bfloat16: "bf16",
            torch.int32: "i32",
            torch.int64: "i64",
        }.get(tensor.dtype)
        if dtype_name is None:
            return
        rank = _tp_rank()
        out = Path(_TRACE_ROOT) / "sglang" / f"rank-{rank}-pid-{os.getpid()}"
        out.mkdir(parents=True, exist_ok=True)
        ordinal = _ordinal
        _ordinal += 1
        safe = name.replace("/", "_")
        filename = f"{ordinal:06d}-{safe}.bin"
        if tensor.dtype == torch.bfloat16:
            payload = tensor.view(torch.uint16).numpy().tobytes(order="C")
        else:
            payload = tensor.numpy().tobytes(order="C")
        (out / filename).write_bytes(payload)
        record = {
            "runtime": "sglang",
            "ordinal": ordinal,
            "name": name,
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "strides": list(tensor.stride()),
            "byte_order": "little",
            "bytes": len(payload),
            "file": filename,
            "tp_rank": rank,
            "tp_size": _safe_int(os.environ.get("WORLD_SIZE"), 4),
        }
        with (out / "manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        _seen.add(name)

    def _record_target(name: str, value) -> None:
        """Retain one last-token target boundary from the audited prefill."""
        if not _armed() or name in _target_trace:
            return
        tensor = _tensor(value)
        if tensor is None or tensor.numel() == 0:
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        matrix = tensor.detach().reshape(-1, tensor.shape[-1])
        # The DFlash prefill may reorder or append speculative rows. The model
        # forward wrapper resolves prompt position 999 to its actual row before
        # any layer hook records a boundary.
        if _target_row < 0 or matrix.shape[0] <= _target_row:
            return
        row = matrix[_target_row]
        _target_trace_dtypes[name] = 32 if row.dtype == torch.float32 else 16
        _target_trace[name] = row.to(torch.float16).clone()

    # Trace the target Qwen3.5 trajectory before DFlash context projection.
    # These are external in-memory hooks; the SGLang source remains read-only.
    import sglang.srt.layers.communicator as _communicator
    import sglang.srt.models.qwen3_5 as _q35

    _orig_target_model_forward = _q35.Qwen3_5Model.forward

    def _traced_target_model_forward(self, *args, **kwargs):
        global _target_row
        if _armed() and _target_row < 0:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            if isinstance(positions, torch.Tensor) and positions.numel():
                host_positions = positions.detach().reshape(-1).cpu().tolist()
                try:
                    matches = [
                        index
                        for index, position in enumerate(host_positions)
                        if int(position) == _target_position
                    ]
                except (TypeError, ValueError) as error:
                    raise RuntimeError("DFLASH target trace received invalid positions") from error
                if len(matches) != 1:
                    raise RuntimeError(
                        f"DFLASH target trace expected one position {_target_position}, got {len(matches)}"
                    )
                _target_row = matches[0]
                token_id = -1
                if isinstance(input_ids, torch.Tensor) and input_ids.numel() > _target_row:
                    try:
                        token_id = int(input_ids.detach().reshape(-1)[_target_row].cpu().item())
                    except (TypeError, ValueError, RuntimeError) as error:
                        raise RuntimeError("DFLASH target trace could not read the frontier token") from error
                print(
                    f"SGLANG_DFLASH_TARGET_FRONTIER position={_target_position} "
                    f"row={_target_row} token_id={token_id} rows={len(host_positions)}",
                    flush=True,
                )
        return _orig_target_model_forward(self, *args, **kwargs)

    _q35.Qwen3_5Model.forward = _traced_target_model_forward

    def _tag_target_layer(original_init):
        def wrapped(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            layer_id = _safe_int(getattr(self, "layer_id", -1), -1)
            if layer_id < 0:
                return
            self.layer_communicator._dflash_target_layer_id = layer_id
            branch = getattr(self, "linear_attn", None)
            if branch is not None:
                branch.register_forward_hook(
                    lambda _m, _a, out, i=layer_id: _record_target(f"target.layer{i}.branch", out)
                )
            else:
                self.o_proj.register_forward_hook(
                    lambda _m, _a, out, i=layer_id: _record_target(f"target.layer{i}.branch", out)
                )
            self.mlp.register_forward_hook(
                lambda _m, _a, out, i=layer_id: _record_target(f"target.layer{i}.mlp_output", out)
            )

        return wrapped

    _q35.Qwen3_5LinearDecoderLayer.__init__ = _tag_target_layer(
        _q35.Qwen3_5LinearDecoderLayer.__init__
    )
    _q35.Qwen3_5AttentionDecoderLayer.__init__ = _tag_target_layer(
        _q35.Qwen3_5AttentionDecoderLayer.__init__
    )

    _orig_target_prepare_attn = _communicator.LayerCommunicator.prepare_attn

    def _traced_target_prepare_attn(self, hidden_states, residual, forward_batch, *args, **kwargs):
        layer_id = getattr(self, "_dflash_target_layer_id", None)
        if layer_id == 0:
            _record_target("target.input.embedding", hidden_states)
        hidden_states, residual = _orig_target_prepare_attn(
            self, hidden_states, residual, forward_batch, *args, **kwargs
        )
        if layer_id == 0:
            _record_target("target.layer0.attn_norm", hidden_states)
        if layer_id is not None and 0 < layer_id <= 6:
            previous = layer_id - 1
            _record_target(f"target.layer{previous}.output_residual", residual)
            _record_target(f"target.layer{previous}.next_attn_norm", hidden_states)
        return hidden_states, residual

    _communicator.LayerCommunicator.prepare_attn = _traced_target_prepare_attn

    _orig_target_prepare_mlp = _communicator.LayerCommunicator.prepare_mlp

    def _traced_target_prepare_mlp(self, hidden_states, residual, forward_batch, *args, **kwargs):
        hidden_states, residual = _orig_target_prepare_mlp(
            self, hidden_states, residual, forward_batch, *args, **kwargs
        )
        layer_id = getattr(self, "_dflash_target_layer_id", None)
        if layer_id is not None and layer_id < 6:
            _record_target(f"target.layer{layer_id}.post_attn_residual", residual)
            _record_target(f"target.layer{layer_id}.mlp_norm", hidden_states)
        return hidden_states, residual

    _communicator.LayerCommunicator.prepare_mlp = _traced_target_prepare_mlp

    _orig_init = _df.DFlashDraftModel.__init__

    def _traced_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.fc.register_forward_hook(lambda _m, _a, out: _dump("context.fc", out, last_row=True))
        self.hidden_norm.register_forward_hook(lambda _m, _a, out: _dump("context.norm", out, last_row=True))
        self.norm.register_forward_hook(lambda _m, _a, out: _dump("block.final_norm", out))
        for index, layer in enumerate(self.layers):
            layer._dflash_trace_name = f"layer{index}"
            layer.register_forward_pre_hook(
                lambda module, args, i=index: (
                    (
                        _dump(f"layer{i}.input.hidden_pre_norm", args[1]),
                        _dump(f"layer{i}.input.residual", args[3]) if len(args) > 3 else None,
                    )
                    and None
                )
            )
            layer.register_forward_hook(
                lambda _m, _a, out, i=index: (
                    (
                        _dump(f"layer{i}.output.hidden", out[0]),
                        _dump(f"layer{i}.output.residual", out[1]),
                    )
                    and None
                )
            )
            layer.input_layernorm.register_forward_hook(
                lambda _m, _a, out, i=index: _dump(f"layer{i}.attention.norm_output", out)
            )
            layer.self_attn.register_forward_hook(
                lambda _m, _a, out, i=index: _dump(f"layer{i}.attention.wo_reduced", out)
            )
            layer.post_attention_layernorm.register_forward_hook(
                lambda _m, _a, out, i=index: _dump(f"layer{i}.mlp.norm_output", out)
            )
            layer.mlp.register_forward_hook(lambda _m, _a, out, i=index: _dump(f"layer{i}.mlp.w2_reduced", out))
            if layer.attention_conv is not None:
                layer.attention_conv._dflash_trace_name = f"layer{index}.attention"
            if layer.mlp_conv is not None:
                layer.mlp_conv._dflash_trace_name = f"layer{index}.mlp"

    _df.DFlashDraftModel.__init__ = _traced_init

    _orig_project = _df.DFlashDraftModel.project_target_hidden

    def _traced_project(self, target_hidden):
        order = ["target.input.embedding", "target.layer0.attn_norm"]
        for layer_id in range(6):
            order.extend(
                [
                    f"target.layer{layer_id}.branch",
                    f"target.layer{layer_id}.post_attn_residual",
                    f"target.layer{layer_id}.mlp_norm",
                    f"target.layer{layer_id}.mlp_output",
                    f"target.layer{layer_id}.output_residual",
                    f"target.layer{layer_id}.next_attn_norm",
                ]
            )
        if all(name in _target_trace for name in order):
            _dump("target.trajectory", torch.stack([_target_trace[name] for name in order]))
            dtype_codes = [_target_trace_dtypes[name] for name in order]
            dtype_tensor = torch.tensor(dtype_codes, dtype=torch.int32, device=target_hidden.device)
            _dump("target.trajectory_dtypes", dtype_tensor)
        _dump("target.post_layer_residual", target_hidden, last_row=True)
        return _orig_project(self, target_hidden)

    _df.DFlashDraftModel.project_target_hidden = _traced_project

    _orig_forward = _df.DFlashDraftModel.forward

    def _traced_forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        input_count = _safe_int(input_ids.numel(), -1) if input_ids is not None else -1
        if input_count == _safe_int(self.block_size, -2):
            _dump("block.ids", input_ids)
            _dump("block.positions", positions)
            _dump("block.embedding", input_embeds)
        return _orig_forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    _df.DFlashDraftModel.forward = _traced_forward

    _orig_prepare = _df.DFlashGroupedConv.prepare

    def _traced_prepare(self, hidden_states):
        output, delta = _orig_prepare(self, hidden_states)
        name = getattr(self, "_dflash_trace_name", "")
        if name:
            _dump(f"{name}.conv_input", hidden_states)
            _dump(f"{name}.conv_side0", output)
            _dump(f"{name}.conv_delta", delta.reshape(delta.shape[0], -1))
        return output, delta

    _df.DFlashGroupedConv.prepare = _traced_prepare

    _orig_finish = _df.DFlashGroupedConv.finish

    def _traced_finish(self, hidden_states, coefficients):
        output = _orig_finish(self, hidden_states, coefficients)
        name = getattr(self, "_dflash_trace_name", "")
        if name:
            _dump(f"{name}.pre_conv_side1", hidden_states)
            _dump(f"{name}.conv_side1", output)
        return output

    _df.DFlashGroupedConv.finish = _traced_finish

    _orig_candidates = _df.DFlash2DraftModel.compute_candidates

    def _traced_candidates(self, hidden):
        _dump("selector.hidden", hidden)
        ids, unary = _orig_candidates(self, hidden)
        _dump("selector.candidate_ids", ids)
        _dump("selector.unary_scores", unary)
        return ids, unary

    _df.DFlash2DraftModel.compute_candidates = _traced_candidates

    _orig_lattice = _df.CandidateSelector.build_lattice

    def _traced_lattice(self, **kwargs):
        scores = _orig_lattice(self, **kwargs)
        _dump("selector.score_lattice", scores)
        return scores

    _df.CandidateSelector.build_lattice = _traced_lattice

    _orig_sample = _df.CandidateSelector.sample_path

    def _traced_sample(self, **kwargs):
        tokens, q_rows = _orig_sample(self, **kwargs)
        _dump("selector.selected_ids", tokens)
        _dump("selector.realized_probabilities", q_rows)
        return tokens, q_rows

    _df.CandidateSelector.sample_path = _traced_sample

    # DFlash's inner Python hooks run only while CUDA graphs are captured, not
    # when a real request replays them. Flush the retained graph-owned tensor
    # references immediately after the first armed draft-model replay. This is
    # deliberately before target verification/acceptance: the pinned SGLang
    # image has an independent V100 accept-kernel failure, but the first draft
    # block is already complete and is the parity artifact we need.
    import sglang.srt.speculative.dflash_worker_v2 as _dw

    _orig_worker_init = _dw.DFlashWorkerV2.__init__

    def _traced_worker_init(self, *args, **kwargs):
        _orig_worker_init(self, *args, **kwargs)
        # The pinned V100 image's standalone Triton prepare/accept helpers are
        # broken: prepare returns without writing the live bonus/mask IDs or
        # positions, and accept later raises an illegal-memory-access error.
        # Use SGLang's own eager fallback orchestration so the unmodified
        # draft model and CUDA graph receive the audited request tensors.
        self._use_triton_prepare_block = False
        self._use_triton_accept_bonus = False
        original_forward = self.draft_model_runner.forward

        def _traced_draft_forward(forward_batch, *forward_args, **forward_kwargs):
            global _graph_flushed
            if _armed():
                # These are the live inputs staged by DFlashWorkerV2 for this
                # exact audited replay, not CUDA-graph capture placeholders.
                # Remove any same-name startup artifact so the manifest's last
                # record is guaranteed to describe this request.
                for name, value in (
                    ("block.ids", getattr(forward_batch, "input_ids", None)),
                    ("block.positions", getattr(forward_batch, "positions", None)),
                    ("block.embedding", getattr(forward_batch, "input_embeds", None)),
                ):
                    _seen.discard(name)
                    _dump(name, value)
            output = original_forward(forward_batch, *forward_args, **forward_kwargs)
            if _armed() and _graph_refs and not _graph_flushed and bool(getattr(output, "can_run_graph", False)):
                _graph_flushed = True
                for name, tensor in _graph_refs.items():
                    _dump(name, tensor)
            return output

        self.draft_model_runner.forward = _traced_draft_forward

    _dw.DFlashWorkerV2.__init__ = _traced_worker_init

    # The pinned overlap scheduler drops the prefill-produced draft input and
    # presents create_idle_input() on the first decode iteration. Preserve the
    # target prefill bonus locally and restore it only when that empty relay is
    # observed. This is trace-harness orchestration; draft-model math remains
    # unchanged and the exact expected anchor is still enforced by the job.
    _orig_worker_forward_generation = _dw.DFlashWorkerV2.forward_batch_generation

    def _traced_worker_forward_generation(self, batch, on_publish=None):
        spec_info = getattr(batch, "spec_info", None)
        if spec_info is None and not batch.forward_mode.is_extend() and not batch.is_extend_in_batch:
            spec_info = _dw.DFlashDraftInputV2.create_idle_input(device=self.device)
            batch.spec_info = spec_info
        bonus = getattr(spec_info, "bonus_tokens", None)
        if spec_info is not None and isinstance(bonus, torch.Tensor) and bonus.numel() == 0:
            batch_size = len(batch.seq_lens)
            # For tensor parity, always force the already-audited LMDeploy
            # anchor so both runtimes execute the identical draft block. The
            # SGLang target produces 1596 for this prompt, which is useful
            # evidence but cannot seed a same-input draft comparison.
            try:
                anchor = int(os.environ.get("SGLANG_PARITY_ANCHOR_ID", "1144"))
            except ValueError as exc:
                raise RuntimeError("invalid SGLANG_PARITY_ANCHOR_ID") from exc
            spec_info.bonus_tokens = torch.full((batch_size,), anchor, dtype=torch.int64, device=batch.seq_lens.device)
        result = _orig_worker_forward_generation(self, batch, on_publish=on_publish)
        next_draft = getattr(result, "next_draft_input", None)
        next_bonus = getattr(next_draft, "bonus_tokens", None)
        if isinstance(next_bonus, torch.Tensor) and next_bonus.numel():
            self._parity_prefill_bonus = next_bonus.detach().clone()
        return result

    _dw.DFlashWorkerV2.forward_batch_generation = _traced_worker_forward_generation
