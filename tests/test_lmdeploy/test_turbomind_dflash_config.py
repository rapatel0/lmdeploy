# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from lmdeploy.messages import TurbomindEngineConfig


def test_dflash2_config_contract():
    config = TurbomindEngineConfig(
        num_draft_tokens=7,
        speculative_algorithm='dflash2',
        speculative_draft_model='/models/Qwen3.8-27B-DFlash2',
        speculative_dflash_block_size=8,
        speculative_draft_window=2048,
    )

    assert config.speculative_algorithm == 'dflash2'
    assert config.speculative_draft_model.endswith('Qwen3.8-27B-DFlash2')
    assert config.num_draft_tokens == config.speculative_dflash_block_size - 1


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(speculative_algorithm='unknown'),
        dict(speculative_algorithm='dflash2'),
        dict(
            speculative_algorithm='dflash2',
            speculative_draft_model='/draft',
            speculative_dflash_block_size=8,
            num_draft_tokens=4,
        ),
    ],
)
def test_invalid_dflash2_config(kwargs):
    with pytest.raises((AssertionError, ValueError)):
        TurbomindEngineConfig(**kwargs)
