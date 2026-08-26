# Copyright (c) OpenMMLab. All rights reserved.

import _turbomind as _tm

from ..linear import Linear
from ._base import Builder

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
        """Commit the fc projection, replicated on every rank.

        fc consumes the two normalised branches concatenated (hidden*2) and
        projects back to hidden. It is deliberately NOT sharded.

        Its output feeds the draft decoder layer's attention, which is itself
        tensor-parallel and expects the full hidden vector on every rank -- the
        same position the target's layers are in after their own input norm.
        Sharding fc would hand each rank a slice of the feature axis and the
        attention would then read a partial vector as if it were whole.

        Passing no split side is what makes this replicated: _add_linear only
        divides a dimension when a split side is given. This builder also never
        sets self.tp, so it stays ParallelGroup(1), and a split side would
        divide by one and silently produce a replicated weight anyway. Saying
        None states the intent instead of relying on that coincidence.
        """
        self._add_linear('fc', fc)
