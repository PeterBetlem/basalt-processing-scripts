from __future__ import annotations

import numpy as np
import pytest

from basalt_processing.porosity import calculate_porosity


def test_calculate_porosity_uses_the_cylinder_denominator() -> None:
    pores = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]]])
    mask = np.array([[[1, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8)

    result = calculate_porosity(pores, mask)

    assert result.pore_voxels == 2
    assert result.mask_voxels == 3
    assert result.total_fraction == pytest.approx(2 / 3)
    np.testing.assert_allclose(result.by_z, [0.5, 1.0])


def test_calculate_porosity_returns_nan_for_an_empty_mask_slice() -> None:
    result = calculate_porosity(np.ones((2, 1, 1)), np.array([[[1]], [[0]]], dtype=np.uint8))

    assert result.by_z[0] == 1.0
    assert np.isnan(result.by_z[1])


def test_calculate_porosity_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_porosity(np.ones((1, 1, 1)), np.ones((1, 1, 2)))
