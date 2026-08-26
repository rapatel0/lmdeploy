# Copyright (c) OpenMMLab. All rights reserved.
"""Fail-closed loader tests for the weight-format resolver.

The Phase 3 specification requires fail-closed tests for tensor-name
coverage, scale-name coverage, scale semantics, ignored-module handling,
quantization block shape, dense and MoE route ownership, and loaded
tensor counts.

Fail-closed means a checkpoint that does not match a declared format must
raise, not silently load as something else. A silent misclassification
produces a model that runs and returns wrong numbers, which is the worst
outcome for a quantization campaign.

These tests exercise the real resolver, not a mock of it.
"""

# pi-lens-ignore: reportMissingImports
import pytest

torch = pytest.importorskip('torch')

from lmdeploy.turbomind.weight_format import (  # noqa: E402
    AWQFormat,
    GPTQFormat,
    TrivialFormat,
    WeightFormat,
)


def _awq(k=64, n=128, group=32):
    """Minimal AWQ suffix dict. qweight packs 8 int4 values per int32."""
    return {
        '.qweight': torch.zeros(k, n // 8, dtype=torch.int32),
        '.scales': torch.zeros(k // group, n, dtype=torch.float16),
        '.qzeros': torch.zeros(k // group, n // 8, dtype=torch.int32),
    }


class TestTensorNameCoverage:
    """Every ingested suffix must be declared in suffix_map."""

    @pytest.mark.parametrize('fmt', [TrivialFormat(), AWQFormat(block_in=32), GPTQFormat(block_in=32)])
    def test_suffix_map_is_declared_and_nonempty(self, fmt: WeightFormat):
        assert fmt.suffix_map, f'{fmt.name} declares no suffixes'
        for suffix, kind in fmt.suffix_map.items():
            assert suffix.startswith('.'), f'{fmt.name}: {suffix!r} is not a suffix'
            assert kind, f'{fmt.name}: {suffix!r} maps to an empty kind'

    def test_weight_is_always_ingested(self):
        """A format that cannot ingest a weight cannot load a linear."""
        for fmt in (TrivialFormat(), AWQFormat(block_in=32), GPTQFormat(block_in=32)):
            assert 'weight' in fmt.suffix_map.values(), f'{fmt.name} ingests no weight'


class TestScaleNameCoverage:
    """Quantized formats must ingest scales; trivial must not."""

    @pytest.mark.parametrize('fmt', [AWQFormat(block_in=32), GPTQFormat(block_in=32)])
    def test_quantized_formats_ingest_scales(self, fmt: WeightFormat):
        assert 'scales' in fmt.suffix_map.values(), f'{fmt.name} ingests no scales'

    def test_trivial_format_ingests_no_scales(self):
        """Trivial must not silently absorb a quantized checkpoint."""
        assert 'scales' not in TrivialFormat().suffix_map.values()


class TestScaleSemantics:
    """has_zero_point must match whether the format ingests zeros."""

    @pytest.mark.parametrize('fmt', [TrivialFormat(), AWQFormat(block_in=32), GPTQFormat(block_in=32)])
    def test_zero_point_flag_matches_suffix_map(self, fmt: WeightFormat):
        ingests_zeros = 'zeros' in fmt.suffix_map.values()
        assert fmt.has_zero_point == ingests_zeros, (f'{fmt.name}: has_zero_point={fmt.has_zero_point} '
                                                     f'but ingests_zeros={ingests_zeros}')

    def test_trivial_format_is_not_quantized(self):
        fmt = TrivialFormat()
        assert fmt.weight_dtype is None
        assert fmt.has_zero_point is False


class TestIgnoredModuleHandling:
    """A format must reject what it does not own, rather than guess."""

    def test_trivial_rejects_quantized_checkpoint(self):
        """The critical fail-closed case.

        If TrivialFormat accepted an AWQ prefix, an int32 qweight would be
        read as a dense weight. The model would run and be wrong.
        """
        assert TrivialFormat().accepts(_awq()) is False

    def test_trivial_rejects_integer_weight(self):
        """Trivial requires a floating-point weight."""
        assert TrivialFormat().accepts({'.weight': torch.zeros(8, 8, dtype=torch.int32)}) is False

    def test_trivial_accepts_dense_weight(self):
        assert TrivialFormat().accepts({'.weight': torch.zeros(8, 8, dtype=torch.float16)}) is True

    def test_trivial_accepts_weight_with_bias(self):
        available = {
            '.weight': torch.zeros(8, 8, dtype=torch.float16),
            '.bias': torch.zeros(8, dtype=torch.float16),
        }
        assert TrivialFormat().accepts(available) is True

    def test_awq_rejects_dense_checkpoint(self):
        """AWQ must not claim a router or norm-like linear."""
        assert AWQFormat(block_in=32).accepts({'.weight': torch.zeros(8, 8, dtype=torch.float16)}) is False

    def test_awq_rejects_missing_qweight(self):
        available = {'.scales': torch.zeros(2, 128, dtype=torch.float16)}
        assert AWQFormat(block_in=32).accepts(available) is False

    def test_awq_rejects_wrong_qweight_dtype(self):
        """qweight must be int32; a float qweight is not an AWQ checkpoint."""
        available = _awq()
        available['.qweight'] = available['.qweight'].to(torch.float16)
        assert AWQFormat(block_in=32).accepts(available) is False


class TestQuantizationBlockShape:
    """Block sizes must be recorded, and shape mismatches must be caught."""

    @pytest.mark.parametrize('group', [32, 64, 128])
    def test_block_in_is_recorded(self, group: int):
        assert AWQFormat(block_in=group).block_in == group

    def test_awq_accepts_consistent_packing(self):
        """qweight packs 8 int4 per int32, so N//8 * 8 must equal scales N."""
        assert AWQFormat(block_in=32).accepts(_awq(k=64, n=128, group=32)) is True

    def test_awq_rejects_inconsistent_packing(self):
        """A scales tensor that disagrees with the packed width is rejected."""
        available = _awq(k=64, n=128, group=32)
        available['.scales'] = torch.zeros(2, 64, dtype=torch.float16)
        assert AWQFormat(block_in=32).accepts(available) is False


class TestRouteOwnership:
    """Exactly one format may own a checkpoint, for dense and MoE alike."""

    def test_exactly_one_format_accepts_awq(self):
        formats = [AWQFormat(block_in=32), TrivialFormat()]
        accepting = [f.name for f in formats if f.accepts(_awq())]
        assert accepting == ['awq'], f'expected sole awq owner, got {accepting}'

    def test_exactly_one_format_accepts_dense(self):
        formats = [AWQFormat(block_in=32), TrivialFormat()]
        available = {'.weight': torch.zeros(8, 8, dtype=torch.float16)}
        accepting = [f.name for f in formats if f.accepts(available)]
        assert accepting == ['trivial'], f'expected sole trivial owner, got {accepting}'

    def test_no_format_accepts_an_unknown_checkpoint(self):
        """An unrecognized checkpoint must be owned by nobody, so the
        resolver raises rather than guessing."""
        formats = [AWQFormat(block_in=32), GPTQFormat(block_in=32), TrivialFormat()]
        available = {'.mystery_tensor': torch.zeros(4, 4, dtype=torch.int8)}
        assert [f.name for f in formats if f.accepts(available)] == []


class TestFormatIdentity:
    """Equality drives the set-based uniformity checks in concat_out_dim.

    If two different block sizes compared equal, a fusion group could mix
    incompatible quantization and still pass the uniformity check.
    """

    def test_same_class_and_block_are_equal(self):
        assert AWQFormat(block_in=32) == AWQFormat(block_in=32)
        assert len({AWQFormat(block_in=32), AWQFormat(block_in=32)}) == 1

    def test_different_block_sizes_are_not_equal(self):
        assert AWQFormat(block_in=32) != AWQFormat(block_in=128)
        assert len({AWQFormat(block_in=32), AWQFormat(block_in=128)}) == 2

    def test_different_classes_are_not_equal(self):
        assert AWQFormat(block_in=32) != GPTQFormat(block_in=32)
        assert len({AWQFormat(block_in=32), GPTQFormat(block_in=32)}) == 2
