# Copyright (c) OpenMMLab. All rights reserved.
"""DeltaNet weight loading builder and GDN input-projection fusion helpers.

Provides ``DeltaNetBuilder`` for committing DeltaNet weights (GDN input
projections, scalar params, conv1d) and helper functions ``split_qkv``
and ``fuse_gdn``.
"""
from __future__ import annotations

import torch

from ..linear import _dequant_linear, dequant_mixed, transform_output_dim
from ._base import Builder, ParallelGroup, SplitSide


# Weight formats that must be dequantized before the GDN output projection
# reaches the GEMM dispatcher.
#
# Only fp8 is listed. Restricting the set keeps the change narrow:
#
# - trivial already round-trips as identity, so listing it would add nothing.
# - gptq and mxfp4 inherit the base-class dequant, which raises. Dequantizing
#   them unconditionally would convert a currently working path into a crash.
# - awq and compressed-tensors implement dequant, but they have working SM70
#   kernels registered in sm70_884_4.cu, so they must stay quantized.
_GDN_OUT_PROJ_DEQUANT_FORMATS = frozenset({'fp8'})


def _split_qkv_tensor(t: torch.Tensor, num_k_heads: int,
                      num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a tensor along its output dim into Q, K, V via block-level view.

    Views the output dim as ``(total_blocks, -1)`` where the inner block
    size self-adapts: head_dim for weight, head_dim/block_out for scales.
    Splits on the blocks axis, flattens back.
    """
    total_blocks = num_k_heads * 2 + num_v_heads
    t = t.unflatten(-1, (total_blocks, -1))
    q, k, v = t.split([num_k_heads, num_k_heads, num_v_heads], dim=-2)
    return q.flatten(-2, -1), k.flatten(-2, -1), v.flatten(-2, -1)


@transform_output_dim
def split_qkv(t: torch.Tensor, num_k_heads: int,
              num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-level: ``@transform_output_dim`` iterates tensor kinds,
    passing each through ``_split_qkv_tensor``."""
    return _split_qkv_tensor(t, num_k_heads, num_v_heads)


def _fuse_tp_interleave(*tensors: torch.Tensor, tp: int) -> torch.Tensor:
    """Fuse N tensors into one with TP interleaving.

    Reshapes each tensor's output dim as ``(tp, -1)``, concatenates,
    then unrolls ``tp`` back into the output dim.  For tp=1 this
    round-trips to a simple concat.
    """
    parts = [t.view(t.size(0), tp, -1) for t in tensors]
    merged = torch.cat(parts, dim=-1)
    return merged.view(-1, merged.size(-1) * tp)


@transform_output_dim
def fuse_gdn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, *,
             tp: int) -> torch.Tensor:
    """Fuse GDN input projections with TP interleaving.

    Layout per tp-shard: [Q | K | V | Z | B | A].
    ``@transform_output_dim`` handles 1-D bias and ``None`` passthrough.
    """
    tensors = [t for t in (q, k, v, z, b, a) if t is not None]
    return _fuse_tp_interleave(*tensors, tp=tp)


# ---------------------------------------------------------------------------
# DeltaNetBuilder -- Gated Delta Net input projections, scalar params, conv1d
# ---------------------------------------------------------------------------


class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def __init__(self, config, ctx, tp: ParallelGroup):
        super().__init__(config, ctx)
        self.tp = tp
        self.config.tp_size = tp.size

    def add_input_projections(self, *, in_proj_qkv, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None):
        """Fuse GDN input projections via pipeline, commit all linears.

        Pipeline: split_qkv -> dequant_mixed -> fuse_gdn -> commit.

        ``out_proj`` is dequantized to the model data type. There is no
        SM70 GEMM kernel for a block-quantized FP8 operand in this position.
        Every SM70 ``Config_E4M3`` tile declares ``Operand_B_Pack``, so it
        requires a pre-packed B operand, and ``Kernel::is_feasible`` compares
        ``pack_b`` exactly. A checkpoint that quantizes ``out_proj`` while
        listing the input projections in ``modules_to_not_convert``, as
        Qwen3.8-27B-FP8 does, therefore aborts in the first forward pass with
        "No feasible kernel found".

        Dequantizing here keeps the treatment of the output projection
        consistent with the input projections above, which already pass
        through ``dequant_mixed``.
        """
        q, k, v = split_qkv(in_proj_qkv,
                            num_k_heads=self.config.num_k_heads,
                            num_v_heads=self.config.num_v_heads)
        q, k, v, z, b, a = dequant_mixed(q, k, v, in_proj_z, in_proj_b, in_proj_a,
                                           data_type=self.config.data_type)
        fused = fuse_gdn(q, k, v, z, b, a, tp=self.tp.size)
        self._add_linear('in_proj_all', fused, SplitSide.OUTPUT)
        if out_proj is not None:
            if out_proj.weight_format.name in _GDN_OUT_PROJ_DEQUANT_FORMATS:
                out_proj = _dequant_linear(out_proj,
                                           data_type=self.config.data_type)
            self._add_linear('out_proj', out_proj, SplitSide.INPUT)

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""
        if a_log is not None:
            self._add_tensor('A_log', a_log, split_side=SplitSide.OUTPUT)
        if dt_bias is not None:
            self._add_tensor('dt_bias', dt_bias, split_side=SplitSide.OUTPUT)

    def add_conv1d(self, conv1d):
        """Transpose HF layout to TM layout, TP-interleave Q/K/V, commit."""
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        conv1d = conv1d.t().contiguous()
        q, k, v = _split_qkv_tensor(conv1d,
                                    self.config.num_k_heads,
                                    self.config.num_v_heads)
        conv1d = _fuse_tp_interleave(q, k, v, tp=self.tp.size)
        self._add_tensor('conv1d', conv1d, split_side=SplitSide.OUTPUT)
