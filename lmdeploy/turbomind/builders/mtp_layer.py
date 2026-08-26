# Copyright (c) OpenMMLab. All rights reserved.

import _turbomind as _tm

from ..linear import Linear
from ._base import Builder, SplitSide

MTPLayerConfig = _tm.MTPLayerConfig


class MTPLayerBuilder(Builder):
    """Builder for the Multi-Token Prediction layer.

    The norm and decoder-layer children are attached by attribute assignment,
    and each name must match a child in ``MTP_LAYER_WEIGHT_CHILDREN`` on the
    C++ side: ``pre_fc_norm_embedding``, ``pre_fc_norm_hidden``, ``fc``,
    ``decoder_layer`` and ``final_norm``.

    ``fc`` is the exception and must go through :meth:`add_fc`. Assigning a
    resolved ``Linear`` to ``m.fc`` looks like it works -- the object is
    attached and the layer reports as loaded -- but only ``_add_linear``
    builds the per-GPU ``LinearWeight`` and fills in ``input_dim`` and
    ``output_dim``. A directly assigned ``fc`` therefore reaches C++ with
    ``input_dim == 0``, and the predictor derives its hidden size from exactly
    that field.
    """

    def add_fc(self, fc: Linear):
        """Commit the fc projection.

        fc consumes the two normalised branches concatenated, so its input is
        hidden*2 and its output is hidden. Under tensor parallelism the
        concatenated input is the dimension that shards, matching the o_proj
        and down_proj convention, so it splits on the input side.
        """
        self._add_linear('fc', fc, SplitSide.INPUT)
