# Copyright (c) OpenMMLab. All rights reserved.
"""DFlash2 separate-checkpoint weight loader for TurboMind."""

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    DFlashConvBuilder,
    DFlashConvConfig,
    DFlashSelectorBuilder,
    DFlashSelectorConfig,
    DFlashWeightBuilder,
    DFlashWeightConfig,
    ModuleListBuilder,
    ModuleListConfig,
)
from .qwen3 import Qwen3TextModel


def _required_int(mapping, name):
    try:
        return int(mapping[name])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f'invalid DFlash2 config field: {name}') from e


def _optional_float(value, default):
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f'invalid DFlash2 float value: {value!r}') from e


class DFlash2Model(Qwen3TextModel):
    """Build the five-layer DFlash2 draft without embeddings or lm_head."""

    def __init__(self, cfg, *, resolver):
        super().__init__(cfg, resolver=resolver)
        draft = cfg.dflash_config
        self._block_size = _required_int(draft, 'block_size')
        self._conv_taps = _required_int(draft, 'conv_kernel_size')
        self._conv_group_size = _required_int(draft, 'conv_group_size')
        self._selector_rank = _required_int(draft, 'selector_rank')
        self._selector_top_k = _required_int(draft, 'selector_top_k')
        self._mask_token_id = _required_int(draft, 'mask_token_id')
        self._output_multiplier = _optional_float(
            draft.get('output_multiplier', getattr(cfg, 'output_multiplier', None)), 1.0)
        self._final_logit_softcapping = _optional_float(
            draft.get('final_logit_softcapping', getattr(cfg, 'final_logit_softcapping', None)), 0.0)
        try:
            self._target_layer_ids = tuple(int(i) for i in draft['target_layer_ids'])
            self._attn_cfg.window_size = int(getattr(cfg, 'sliding_window', 0) or 0)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('invalid DFlash2 target layers or sliding window') from e
        self._attn_cfg.causal = bool(getattr(cfg, 'is_causal', False))

    def _conv(self, pfx):
        cfg = DFlashConvConfig()
        cfg.taps = self._conv_taps
        cfg.group_size = self._conv_group_size
        builder = DFlashConvBuilder(cfg, self._ctx)
        builder.add_kernel_projection(self._linear(pfx + 'kernel_projection'))
        builder.add_base_kernel(pfx.get('base_kernel'))
        return builder.build()

    def _selector(self, pfx):
        cfg = DFlashSelectorConfig()
        cfg.state_rank = self._selector_rank
        cfg.top_k = self._selector_top_k
        builder = DFlashSelectorBuilder(cfg, self._ctx)
        builder.add_hidden_projection(self._linear(pfx + 'hidden_projection'))
        builder.add_predecessor_codebook(pfx.get('predecessor_codebook'))
        builder.add_successor_codebook(pfx.get('successor_codebook'))
        return builder.build()

    def _layers_and_convs(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        attention_convs = ModuleListBuilder(ModuleListConfig(), self._ctx)
        mlp_convs = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, layer_pfx in pfx.slices(0, self.cfg.num_hidden_layers):
            layer = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            layer.attention_norm = self.norm(layer_pfx + 'input_layernorm')
            layer.attention = self.attn(layer_pfx + 'self_attn')
            layer.ffn_norm = self.norm(layer_pfx + 'post_attention_layernorm')
            layer.feed_forward = self.ffn(layer_pfx + 'mlp')
            layers[i] = layer.build()
            attention_convs[i] = self._conv(layer_pfx + 'attention_conv')
            mlp_convs[i] = self._conv(layer_pfx + 'mlp_conv')
        return layers.build(), attention_convs.build(), mlp_convs.build()

    def draft(self, pfx):
        cfg = DFlashWeightConfig()
        cfg.block_size = self._block_size
        cfg.draft_window = self._attn_cfg.window_size
        cfg.num_context_features = len(self._target_layer_ids)
        cfg.mask_token_id = self._mask_token_id
        cfg.target_layer_ids = list(self._target_layer_ids)
        cfg.output_multiplier = self._output_multiplier
        cfg.final_logit_softcapping = self._final_logit_softcapping
        builder = DFlashWeightBuilder(cfg, self._ctx)
        builder.add_fc(self._linear(pfx + 'fc'))
        builder.hidden_norm = self.norm(pfx + 'hidden_norm')
        builder.final_norm = self.norm(pfx + 'norm')
        builder.layers, builder.attention_convs, builder.mlp_convs = self._layers_and_convs(pfx + 'layers')
        builder.selector = self._selector(pfx + 'candidate_selector')
        return builder.build()
