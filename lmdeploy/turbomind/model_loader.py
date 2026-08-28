# Copyright (c) OpenMMLab. All rights reserved.
"""ModelLoader: coordinates loading a model's weights into the TurboMind runtime."""
import torch

from .builders._base import Context, ParallelGroup
from .checkpoint import Prefix, create_checkpoint


class ModelLoader:
    """Coordinates loading a model's weights into the TurboMind runtime.

    Holds the model, model_comm handle, and model_path. Extracts GPU topology handles from model_comm and binds them
    onto the model at construction time. Provides export() and export_iter() to load checkpoint weights and commit them
    to the C++ runtime.
    """

    def __init__(self, model, model_comm, gpu_count, model_path,
                 data_type, engine_config):
        self.model = model
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self.data_type = data_type
        self.engine_config = engine_config
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        ctx = Context(
            [mc.context(g) for g in range(self.gpu_count)],
            data_type=self.data_type,
        )
        ec = self.engine_config

        attn_tp = ParallelGroup(ec.attn_tp_size,
                                [mc.attn_tp_rank(g) for g in range(self.gpu_count)])
        mlp_tp = ParallelGroup(ec.mlp_tp_size,
                               [mc.mlp_tp_rank(g) for g in range(self.gpu_count)])
        ep = ParallelGroup(ec.ep,
                           [mc.ep_rank(g) for g in range(self.gpu_count)])
        model_tp = ParallelGroup(ec.attn_tp_size * ec.attn_cp_size,
                                 [mc.model_tp_rank(g) for g in range(self.gpu_count)])

        self._runtime = dict(
            ctx=ctx,
            root_handles=[mc.root(g) for g in range(self.gpu_count)],
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            ep=ep,
            model_tp=model_tp,
        )
        self.model.bind_runtime(**self._runtime)

    def _build_dflash(self, draft_ckpt):
        if not getattr(self.model, 'supports_dflash2', False):
            raise ValueError(f'{type(self.model).__name__} does not support a DFlash2 draft')

        from transformers import AutoConfig

        from .models.dflash import DFlash2Model
        from .weight_format import TrivialFormat, WeightFormatResolver

        draft_path = self.engine_config.speculative_draft_model
        draft_cfg = AutoConfig.from_pretrained(draft_path, trust_remote_code=False)
        if 'DFlash2DraftModel' not in getattr(draft_cfg, 'architectures', []):
            raise ValueError(f'{draft_path} is not a DFlash2DraftModel checkpoint')
        resolver = WeightFormatResolver(
            data_type=self.data_type,
            formats=[TrivialFormat()],
        )
        draft_model = DFlash2Model(draft_cfg, resolver=resolver)
        draft_model.bind_runtime(**self._runtime)
        return draft_model.draft(Prefix(draft_ckpt))

    def _export_model(self):
        ckpt = create_checkpoint(
            self.model_path,
            mappings=getattr(self.model, '_loader_mappings', []))
        draft_ckpt = None
        try:
            if self.engine_config.speculative_algorithm == 'dflash2':
                draft_ckpt = create_checkpoint(self.engine_config.speculative_draft_model)
                dflash = self._build_dflash(draft_ckpt)
                self.model.model(Prefix(ckpt), dflash=dflash)
            else:
                self.model.model(Prefix(ckpt))
        finally:
            if draft_ckpt is not None:
                draft_ckpt.close()
            ckpt.close()

    def export(self):
        self._export_model()
        torch.cuda.empty_cache()

    def export_iter(self):
        self._export_model()
        yield -1
        torch.cuda.empty_cache()
