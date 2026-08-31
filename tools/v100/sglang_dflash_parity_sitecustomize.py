"""Trace-only SGLang DFlash hooks loaded through sitecustomize.

This file monkey-patches the installed reference runtime in memory; it never
changes or ships SGLang source. Set SGLANG_DFLASH_PARITY_DIR to enable it.
"""

from __future__ import annotations

import hashlib
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
    _target_feature_trace: dict[int, torch.Tensor] = {}
    _graph_flushed = False
    _ordinal = 0
    _arm_file = os.environ.get("SGLANG_DFLASH_PARITY_ARM_FILE", "")
    try:
        _capture_block_index = max(1, int(os.environ.get("SGLANG_DFLASH_PARITY_BLOCK_INDEX", "1")))
    except (TypeError, ValueError):
        _capture_block_index = 1
    _draft_blocks_seen = 0
    _capture_enabled = _capture_block_index == 1
    try:
        _target_position = int(os.environ.get("SGLANG_DFLASH_TARGET_POSITION", "999"))
        _target_token_id = int(os.environ.get("SGLANG_DFLASH_TARGET_TOKEN_ID", "-1"))
    except (TypeError, ValueError):
        _target_position = 999
        _target_token_id = -1
    _target_row = -1
    _target_resolved_position = -1
    _target_input_hash = ""
    _target_input_rows = 0

    def _request_armed() -> bool:
        return not _arm_file or Path(_arm_file).exists()

    def _armed() -> bool:
        # SGLang executes synthetic DFlash blocks while warming kernels during
        # server startup. The external client creates this shared marker only
        # after health_generate is ready, immediately before the audited real
        # request. Keeping every hook inert until then prevents a valid-looking
        # trace of dummy token IDs and avoids synchronizing warm-up kernels.
        return _request_armed() and _capture_enabled

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
            # Q/K hook outputs are views that later kernels mutate in place
            # (qkv -> QK norm -> RoPE). Snapshot only those exact arithmetic
            # boundaries. Other graph references (notably block.ids) are
            # intentionally populated later by replay and must stay live.
            immutable_attention = {
                "layer0.attention.qkv_projection",
                "layer0.attention.q_normalized",
                "layer0.attention.k_normalized",
                "layer0.attention.q_rotated",
                "layer0.attention.k_rotated",
                "layer0.attention.core_output",
            }
            _graph_refs[name] = tensor.detach().clone() if name in immutable_attention else tensor.detach()
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
        if name in {"target.trajectory", "target.trajectory_dtypes"}:
            record.update(
                {
                    "position": _target_resolved_position,
                    "token_id": _target_token_id,
                    "resolved_row": _target_row,
                    "input_rows": _target_input_rows,
                    "input_ids_sha256": _target_input_hash,
                    "forward_mode": "target_prefill" if _target_input_rows >= 1000 else "target_verify",
                }
            )
        with (out / "manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        _seen.add(name)

    _orig_apply_qk_norm = _df.apply_qk_norm

    def _traced_apply_qk_norm(q, k, q_norm, k_norm, head_dim):
        q, k = _orig_apply_qk_norm(q, k, q_norm, k_norm, head_dim)
        _dump("layer0.attention.q_normalized", q)
        _dump("layer0.attention.k_normalized", k)
        return q, k

    _df.apply_qk_norm = _traced_apply_qk_norm

    try:
        import sglang.srt.speculative.triton_ops.fused_kv_materialize as _fused_kv

        _orig_fused_materialize = _fused_kv.FusedKVMaterializeHelper.materialize

        def _traced_fused_materialize(self, ctx_hidden, positions, write_layer_kv):
            rows = int(ctx_hidden.shape[0])
            scope = "prompt" if rows == 1000 else "frontier"

            def _traced_write(layer_idx, cache_k, cache_v):
                _dump(f"context.{scope}.layer{layer_idx}.cache_k", cache_k)
                _dump(f"context.{scope}.layer{layer_idx}.cache_v", cache_v)
                write_layer_kv(layer_idx, cache_k, cache_v)

            result = _orig_fused_materialize(self, ctx_hidden, positions, _traced_write)
            if self._proj_workspace is not None:
                projected = self._proj_workspace[:rows].view(rows, self.n_layers, self.layer_out_dim)
                for layer_idx in range(self.n_layers):
                    _dump(f"context.{scope}.layer{layer_idx}.k_projection", projected[:, layer_idx, : self.kv_size])
                    _dump(f"context.{scope}.layer{layer_idx}.v_projection", projected[:, layer_idx, self.kv_size :])
            return result

        _fused_kv.FusedKVMaterializeHelper.materialize = _traced_fused_materialize
    except (AttributeError, ImportError):
        pass

    def _target_order() -> list[str]:
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
        return order

    def _record_target(name: str, value) -> None:
        """Retain one target boundary from the requested live position/token."""
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
        print(
            f"SGLANG_DFLASH_TARGET_BOUNDARY name={name} row={_target_row} rows={matrix.shape[0]} dtype={row.dtype}",
            flush=True,
        )
        order = _target_order()
        if all(boundary in _target_trace for boundary in order):
            _dump("target.trajectory", torch.stack([_target_trace[boundary] for boundary in order]))
            dtype_codes = [_target_trace_dtypes[boundary] for boundary in order]
            _dump("target.trajectory_dtypes", torch.tensor(dtype_codes, dtype=torch.int32, device=row.device))

    def _record_target_detail(name: str, value) -> None:
        """Persist a varying-width target row outside the stacked trajectory."""
        if not _armed() or name in _seen:
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
        if _target_row < 0 or matrix.shape[0] <= _target_row:
            return
        row = matrix[_target_row].to(torch.float16).clone().reshape(1, -1)
        _dump(name, row)
        print(
            f"SGLANG_DFLASH_TARGET_DETAIL name={name} row={_target_row} width={row.shape[-1]}",
            flush=True,
        )

    def _record_target_feature(layer_id: int, value) -> None:
        """Retain audited prompt residuals consumed by DFlash ProjectContext."""
        feature_ids = (5, 19, 33, 47, 61)
        if layer_id not in feature_ids or layer_id in _target_feature_trace:
            return
        tensor = _tensor(value)
        if not _armed() or tensor is None or tensor.numel() == 0:
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        matrix = tensor.detach().reshape(-1, tensor.shape[-1])
        if _target_row < 0 or matrix.shape[0] <= _target_row:
            return
        _target_feature_trace[layer_id] = matrix[_target_row].to(torch.float16).clone()
        print(f"SGLANG_DFLASH_TARGET_FEATURE layer={layer_id} row={_target_row}", flush=True)
        if all(feature_id in _target_feature_trace for feature_id in feature_ids):
            _dump(
                "target.prompt_features",
                torch.stack([_target_feature_trace[feature_id] for feature_id in feature_ids]),
            )

    # Trace the target Qwen3.5 trajectory before DFlash context projection.
    # These are external in-memory hooks; the SGLang source remains read-only.
    import sglang.srt.layers.communicator as _communicator
    import sglang.srt.models.qwen3_5 as _q35

    def _resolve_target_frontier(input_ids, positions) -> None:
        global _target_input_hash, _target_input_rows, _target_resolved_position, _target_row
        if not _armed() or _target_row >= 0 or not isinstance(positions, torch.Tensor) or not positions.numel():
            return
        host_positions = positions.detach().reshape(-1).cpu().tolist()
        try:
            matches = (
                list(range(len(host_positions)))
                if _target_position < 0
                else [index for index, position in enumerate(host_positions) if int(position) == _target_position]
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("DFLASH target trace received invalid positions") from error
        if not matches:
            # Chunked prefill and earlier decode calls may not yet contain the
            # requested verification position. Remain armed for the live call.
            return
        # Token-only alignment is reserved for the small merged verification
        # batch; never select a coincidental occurrence from the 1K prefill.
        if _target_position < 0 and len(host_positions) > 64:
            return
        host_ids = None
        if isinstance(input_ids, torch.Tensor) and input_ids.numel():
            try:
                host_ids = input_ids.detach().reshape(-1).cpu().tolist()
            except (TypeError, ValueError, RuntimeError) as error:
                raise RuntimeError("DFLASH target trace could not read frontier tokens") from error
        # Merged scheduler batches can contain the same logical position more
        # than once. Resolve the exact target-verification row by the audited
        # token as well as the position; placeholder-only calls remain armed.
        if _target_token_id >= 0 and host_ids is not None:
            try:
                matches = [
                    index for index in matches if index < len(host_ids) and int(host_ids[index]) == _target_token_id
                ]
            except (TypeError, ValueError) as error:
                raise RuntimeError("DFLASH target trace received invalid frontier tokens") from error
        if not matches:
            return
        # Target verification can broadcast the same bonus token over several
        # merged 8-row blocks. TurboMind retains the final row of its submitted
        # verifier slab, so align to the final exact token match and persist the
        # resolved row/position below rather than assuming scheduler order.
        candidate_row = matches[-1]
        try:
            token_id = int(host_ids[candidate_row]) if host_ids is not None else -1
        except (TypeError, ValueError) as error:
            raise RuntimeError("DFLASH target trace received an invalid selected token") from error
        _target_row = candidate_row
        try:
            resolved_position = int(host_positions[candidate_row])
        except (TypeError, ValueError) as error:
            raise RuntimeError("DFLASH target trace received an invalid selected position") from error
        _target_resolved_position = resolved_position
        _target_input_rows = len(host_positions)
        if host_ids is not None:
            prefix = torch.tensor(host_ids[: min(1000, len(host_ids))], dtype=torch.int32)
            _target_input_hash = hashlib.sha256(prefix.numpy().tobytes(order="C")).hexdigest()
        print(
            f"SGLANG_DFLASH_TARGET_FRONTIER position={resolved_position} "
            f"row={_target_row} token_id={token_id} rows={len(host_positions)}",
            flush=True,
        )

    _orig_target_model_forward = _q35.Qwen3_5ForCausalLM.forward

    def _traced_target_model_forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        _resolve_target_frontier(input_ids, positions)
        return _orig_target_model_forward(self, *args, **kwargs)

    _q35.Qwen3_5ForCausalLM.forward = _traced_target_model_forward

    # ModelRunner.forward is the live target entry even when TorchDynamo has
    # compiled past the Python Qwen model wrapper. Patch the external class
    # method before workers are constructed so the audited row is resolved
    # before target layer hooks execute.
    import sglang.srt.model_executor.model_runner as _model_runner

    _orig_model_runner_forward = _model_runner.ModelRunner.forward

    def _traced_model_runner_forward(self, forward_batch, *args, **kwargs):
        if not bool(getattr(self, "is_draft_worker", False)):
            _resolve_target_frontier(
                getattr(forward_batch, "input_ids", None), getattr(forward_batch, "positions", None)
            )
        return _orig_model_runner_forward(self, forward_batch, *args, **kwargs)

    _model_runner.ModelRunner.forward = _traced_model_runner_forward

    _orig_model_runner_split_prefill = _model_runner.ModelRunner.forward_split_prefill

    def _traced_model_runner_split_prefill(self, forward_batch, *args, **kwargs):
        if not bool(getattr(self, "is_draft_worker", False)):
            _resolve_target_frontier(
                getattr(forward_batch, "input_ids", None), getattr(forward_batch, "positions", None)
            )
        return _orig_model_runner_split_prefill(self, forward_batch, *args, **kwargs)

    _model_runner.ModelRunner.forward_split_prefill = _traced_model_runner_split_prefill

    def _tag_target_layer(original_init):
        def wrapped(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            layer_id = _safe_int(getattr(self, "layer_id", -1), -1)
            if layer_id < 0:
                return
            self.layer_communicator._dflash_target_layer_id = layer_id

            # Some pinned-image communicator subclasses override the base
            # methods patched below. Wrap the concrete per-layer instance so
            # every target decoder path records the exact live boundaries.
            original_prepare_attn = self.layer_communicator.prepare_attn

            def traced_prepare_attn(hidden_states, residual, forward_batch, *call_args, **call_kwargs):
                global _target_position, _target_row
                if layer_id == 0:
                    _resolve_target_frontier(
                        getattr(forward_batch, "input_ids", None), getattr(forward_batch, "positions", None)
                    )
                    if (
                        _target_position == 999
                        and _target_row < 0
                        and isinstance(hidden_states, torch.Tensor)
                        and hidden_states.shape[0] > 999
                    ):
                        _target_position = 999
                        _target_row = 999
                        print(
                            "SGLANG_DFLASH_TARGET_FRONTIER position=999 row=999 "
                            f"token_id=198 rows={hidden_states.shape[0]} source=audited_layer_layout",
                            flush=True,
                        )
                    _record_target("target.input.embedding", hidden_states)
                normalized, next_residual = original_prepare_attn(
                    hidden_states, residual, forward_batch, *call_args, **call_kwargs
                )
                if layer_id == 0:
                    _record_target("target.layer0.attn_norm", normalized)
                if layer_id > 0:
                    previous = layer_id - 1
                    _record_target_feature(previous, next_residual)
                    if layer_id <= 6:
                        _record_target(f"target.layer{previous}.output_residual", next_residual)
                        _record_target(f"target.layer{previous}.next_attn_norm", normalized)
                return normalized, next_residual

            self.layer_communicator.prepare_attn = traced_prepare_attn

            original_prepare_mlp = self.layer_communicator.prepare_mlp

            def traced_prepare_mlp(hidden_states, residual, forward_batch, *call_args, **call_kwargs):
                normalized, next_residual = original_prepare_mlp(
                    hidden_states, residual, forward_batch, *call_args, **call_kwargs
                )
                if layer_id < 6:
                    _record_target(f"target.layer{layer_id}.post_attn_residual", next_residual)
                    _record_target(f"target.layer{layer_id}.mlp_norm", normalized)
                return normalized, next_residual

            self.layer_communicator.prepare_mlp = traced_prepare_mlp

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
            if layer_id == 0:
                gate_up = getattr(self.mlp, "gate_up_proj", None)
                original_fused = getattr(gate_up, "forward_fused_silu_and_mul", None)
                if gate_up is not None and original_fused is not None:

                    def traced_fused_activation(input_):
                        output = original_fused(input_)
                        if output is not None:
                            _record_target_detail("target.layer0.mlp_activation", output)
                        return output

                    gate_up.forward_fused_silu_and_mul = traced_fused_activation
                activation = getattr(self.mlp, "act_fn", None)
                if activation is not None:
                    activation.register_forward_hook(
                        lambda _m, _a, out: _record_target_detail("target.layer0.mlp_activation", out)
                    )

        return wrapped

    _q35.Qwen3_5LinearDecoderLayer.__init__ = _tag_target_layer(_q35.Qwen3_5LinearDecoderLayer.__init__)
    _q35.Qwen3_5AttentionDecoderLayer.__init__ = _tag_target_layer(_q35.Qwen3_5AttentionDecoderLayer.__init__)

    _orig_target_prepare_attn = _communicator.LayerCommunicator.prepare_attn

    def _traced_target_prepare_attn(self, hidden_states, residual, forward_batch, *args, **kwargs):
        global _target_position, _target_row
        layer_id = getattr(self, "_dflash_target_layer_id", None)
        if layer_id == 0:
            # This communicator boundary is reached on every real target
            # prefill path, including split-prefill and compiled ModelRunner
            # variants. Resolve against its live ForwardBatch before recording.
            _resolve_target_frontier(
                getattr(forward_batch, "input_ids", None), getattr(forward_batch, "positions", None)
            )
            if (
                _target_position == 999
                and _target_row < 0
                and isinstance(hidden_states, torch.Tensor)
                and hidden_states.shape[0] > 999
            ):
                # The pinned image does not expose input_ids/positions on its
                # split-prefill ForwardBatch. The audited request is hash-
                # checked and logged as one uncached 1,008-row prefill, so row
                # 999 is the unique prompt frontier before eight placeholders.
                _target_position = 999
                _target_row = 999
                print(
                    "SGLANG_DFLASH_TARGET_FRONTIER position=999 row=999 "
                    f"token_id=198 rows={hidden_states.shape[0]} source=audited_prefill_layout",
                    flush=True,
                )
            _record_target("target.input.embedding", hidden_states)
        hidden_states, residual = _orig_target_prepare_attn(
            self, hidden_states, residual, forward_batch, *args, **kwargs
        )
        if layer_id == 0:
            _record_target("target.layer0.attn_norm", hidden_states)
        if layer_id is not None and layer_id > 0:
            previous = layer_id - 1
            _record_target_feature(previous, residual)
            if layer_id <= 6:
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

    def _trace_context_boundary(name, full_name, output):
        if output.ndim == 2 and output.shape[0] == 1000:
            _dump(full_name, output)
        _dump(name, output, last_row=True)

    def _trace_layer0_rotary(_module, _args, output):
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            _dump("layer0.attention.q_rotated", output[0])
            _dump("layer0.attention.k_rotated", output[1])

    def _traced_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.fc.register_forward_hook(lambda _m, _a, out: _trace_context_boundary("context.fc", "context.full_fc", out))
        self.hidden_norm.register_forward_hook(
            lambda _m, _a, out: _trace_context_boundary("context.norm", "context.full_norm", out)
        )
        self.norm.register_forward_hook(lambda _m, _a, out: _dump("block.final_norm", out))
        for index, layer in enumerate(self.layers):
            layer._dflash_trace_name = f"layer{index}"
            if index == 0:
                layer.self_attn.qkv_proj.register_forward_hook(
                    lambda _m, _a, out: _dump("layer0.attention.qkv_projection", out)
                )
                layer.self_attn.q_norm.register_forward_hook(
                    lambda _m, _a, out: _dump("layer0.attention.q_normalized", out)
                )
                layer.self_attn.k_norm.register_forward_hook(
                    lambda _m, _a, out: _dump("layer0.attention.k_normalized", out)
                )
                layer.self_attn.rotary_emb.register_forward_hook(_trace_layer0_rotary)
                layer.self_attn.attn.register_forward_hook(
                    lambda _m, _a, out: _dump("layer0.attention.core_output", out)
                )
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
        order = _target_order()
        if all(name in _target_trace for name in order):
            _dump("target.trajectory", torch.stack([_target_trace[name] for name in order]))
            dtype_codes = [_target_trace_dtypes[name] for name in order]
            dtype_tensor = torch.tensor(dtype_codes, dtype=torch.int32, device=target_hidden.device)
            _dump("target.trajectory_dtypes", dtype_tensor)
        feature_ids = (5, 19, 33, 47, 61)
        if all(feature_id in _target_feature_trace for feature_id in feature_ids):
            expected = torch.cat([_target_feature_trace[feature_id] for feature_id in feature_ids]).to(
                target_hidden.device
            )
            matrix = target_hidden.detach().reshape(-1, target_hidden.shape[-1])
            if matrix.shape[1] == expected.numel():
                try:
                    row_sums = []
                    for start in range(0, matrix.shape[0], 64):
                        delta = matrix[start : start + 64].float() - expected.float()
                        row_sums.append((delta * delta).mean(dim=1).cpu())
                    row_rms = torch.cat(row_sums).sqrt()
                    best_row = int(torch.argmin(row_rms).item())
                    selected_row = _target_row if 0 <= _target_row < matrix.shape[0] else matrix.shape[0] - 1
                    segment_rms = []
                    selected = matrix[selected_row].reshape(5, 5120).float()
                    expected_segments = expected.reshape(5, 5120).float()
                    for output_index in range(5):
                        segment_rms.append(
                            [
                                float(
                                    torch.sqrt(
                                        torch.mean((selected[output_index] - expected_segments[input_index]) ** 2)
                                    ).item()
                                )
                                for input_index in range(5)
                            ]
                        )
                    report = {
                        "shape": list(matrix.shape),
                        "resolved_row": _target_row,
                        "selected_row": selected_row,
                        "selected_rms": float(row_rms[selected_row].item()),
                        "best_row": best_row,
                        "best_rms": float(row_rms[best_row].item()),
                        "segment_rms": segment_rms,
                    }
                except (RuntimeError, TypeError, ValueError, IndexError) as error:
                    raise RuntimeError(f"cannot compare target projector alignment: {error}") from error
                print(
                    "SGLANG_DFLASH_TARGET_PROJECT_ALIGNMENT " + json.dumps(report, sort_keys=True),
                    flush=True,
                )
        if target_hidden.ndim == 2 and target_hidden.shape[0] == 1000:
            _dump("target.full_context", target_hidden)
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
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput as _DFlashVerifyInput

    # The pinned image omitted the scheduler merge contract from its v1 verify
    # record. Supply the same tensor concatenation semantics in memory so the
    # single audited request can advance from target prefill into verification.
    if not hasattr(_DFlashVerifyInput, "merge_batch"):

        def _merge_verify_batch(self, other):
            self.draft_token = torch.cat([self.draft_token, other.draft_token], dim=0)
            self.positions = torch.cat([self.positions, other.positions], dim=0)
            self.custom_mask = None
            for field in ("selector_candidate_ids", "selector_q_rows"):
                left = getattr(self, field, None)
                right = getattr(other, field, None)
                if left is None or left.numel() == 0:
                    setattr(self, field, right)
                elif right is not None and right.numel() > 0:
                    setattr(self, field, torch.cat([left, right], dim=0))

        _DFlashVerifyInput.merge_batch = _merge_verify_batch

    # The pinned overlap scheduler can present the same audited request as both
    # running_batch and last_batch while transitioning out of target verify.
    # Merging those aliases duplicates one request into batch size two, which
    # then overruns the batch-one selector graph buffers. Ignore only this exact
    # same-request alias; distinct request batches retain production merging.
    import sglang.srt.managers.schedule_batch as _schedule_batch

    _orig_schedule_merge = _schedule_batch.ScheduleBatch.merge_batch

    def _trace_safe_schedule_merge(self, other):
        if isinstance(getattr(self, "spec_info", None), _DFlashVerifyInput):
            left = {getattr(req, "rid", id(req)) for req in getattr(self, "reqs", ())}
            right = {getattr(req, "rid", id(req)) for req in getattr(other, "reqs", ())}
            if left and left == right:
                print(f"SGLANG_DFLASH_SKIP_ALIAS_MERGE requests={sorted(map(str, left))}", flush=True)
                return
        return _orig_schedule_merge(self, other)

    _schedule_batch.ScheduleBatch.merge_batch = _trace_safe_schedule_merge

    _orig_compute_verify = _dw.compute_dflash_correct_drafts_and_bonus

    def _traced_compute_verify(*, candidates, target_predict):
        _dump("verify.candidates", candidates)
        _dump("verify.target_top1", target_predict)
        accept_len, bonus = _orig_compute_verify(candidates=candidates, target_predict=target_predict)
        _dump("verify.accept_len", accept_len)
        _dump("verify.bonus", bonus)
        if _armed():
            print(
                f"SGLANG_DFLASH_VERIFY candidates={candidates.detach().cpu().reshape(-1).tolist()} "
                f"target_top1={target_predict.detach().cpu().reshape(-1).tolist()} "
                f"accept_len={accept_len.detach().cpu().reshape(-1).tolist()} "
                f"bonus={bonus.detach().cpu().reshape(-1).tolist()}",
                flush=True,
            )
        return accept_len, bonus

    _dw.compute_dflash_correct_drafts_and_bonus = _traced_compute_verify

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
            global _capture_enabled, _draft_blocks_seen, _graph_flushed
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
            if _request_armed():
                _draft_blocks_seen += 1
                if _draft_blocks_seen >= _capture_block_index - 1:
                    _capture_enabled = True
            if (
                _armed()
                and _draft_blocks_seen >= _capture_block_index
                and _graph_refs
                and not _graph_flushed
                and bool(getattr(output, "can_run_graph", False))
            ):
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
        if (
            not isinstance(spec_info, _dw.DFlashDraftInputV2)
            and not batch.forward_mode.is_extend()
            and not batch.is_extend_in_batch
        ):
            retained = getattr(self, "_parity_next_draft", None)
            spec_info = (
                retained
                if isinstance(retained, _dw.DFlashDraftInputV2)
                else _dw.DFlashDraftInputV2.create_idle_input(device=self.device)
            )
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
            self._parity_next_draft = next_draft
        return result

    _dw.DFlashWorkerV2.forward_batch_generation = _traced_worker_forward_generation
