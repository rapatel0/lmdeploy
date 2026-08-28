# Copyright (c) OpenMMLab. All rights reserved.

import _turbomind as _tm
import torch

from ..linear import Linear
from ._base import Builder

DFlashConvConfig = _tm.DFlashConvConfig
DFlashSelectorConfig = _tm.DFlashSelectorConfig
DFlashWeightConfig = _tm.DFlashWeightConfig


class DFlashConvBuilder(Builder):
    """Build one replicated dynamic grouped-convolution adapter."""

    def add_kernel_projection(self, projection: Linear):
        self._add_linear('kernel_projection', projection)

    def add_base_kernel(self, tensor: torch.Tensor):
        self._add_tensor('base_kernel', tensor)


class DFlashSelectorBuilder(Builder):
    """Build the replicated DFlash2 candidate-path selector."""

    def add_hidden_projection(self, projection: Linear):
        self._add_linear('hidden_projection', projection)

    def add_predecessor_codebook(self, tensor: torch.Tensor):
        self._add_tensor('predecessor_codebook', tensor)

    def add_successor_codebook(self, tensor: torch.Tensor):
        self._add_tensor('successor_codebook', tensor)


class DFlashWeightBuilder(Builder):
    """Build the separate DFlash2 draft root.

    Embeddings and the vocabulary projection remain target-owned. The context
    projection is replicated because every TP rank enters each decoder layer
    with the complete hidden vector.
    """

    def add_fc(self, projection: Linear):
        self._add_linear('fc', projection)
