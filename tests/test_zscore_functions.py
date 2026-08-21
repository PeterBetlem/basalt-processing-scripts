import numpy as np

from basalt_processing.zscore_functions import zscore_volume_using_matrix_stats


def test_zscore_volume_uses_one_cells_in_binary_mask_for_matrix_statistics():
    volume = np.array([10.0, 14.0, 100.0, np.nan], dtype=np.float32)
    pore_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    zscore, stats = zscore_volume_using_matrix_stats(volume, pore_mask)

    assert stats == {"mean_matrix": 12.0, "std_matrix": 2.0, "n_matrix": 2}
    np.testing.assert_allclose(zscore[:3], [-1.0, 1.0, 44.0])
    assert np.isnan(zscore[3])
