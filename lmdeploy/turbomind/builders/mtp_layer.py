# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder

MTPLayerConfig = _tm.MTPLayerConfig


class MTPLayerBuilder(Builder):
    """Pure container builder for the Multi-Token Prediction layer.

    Children are attached by attribute assignment, and each name must match a
    child in ``MTP_LAYER_WEIGHT_CHILDREN`` on the C++ side:
    ``pre_fc_norm_embedding``, ``pre_fc_norm_hidden``, ``fc``,
    ``decoder_layer`` and ``final_norm``.
    """
    pass
