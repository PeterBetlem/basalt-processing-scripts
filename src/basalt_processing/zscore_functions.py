import numpy as np
from typing import Union

try:
    import h5py
    H5Like = Union["h5py.Dataset", np.ndarray]
except Exception:
    H5Like = Union[np.ndarray]


def zscore_volume_using_matrix_stats(
    dset: H5Like,
    pore_mask: H5Like,
    dtype=np.float32,
):
    """
    Z-score a 3D volume using mean/std computed from matrix voxels only.

    Mask convention:
      pore_mask == 1     -> matrix
      pore_mask == 0     -> pore space (and possibly outside sample)

    Outside-sample voxels are excluded by the NaNs in ``dset``.

    Z-score is applied everywhere in the volume.
    """

    # Load arrays
    v = np.asarray(dset, dtype=dtype)

    # Current threshold masks are binary (1=matrix, 0=pore/outside).
    matrix = np.asarray(pore_mask)
    matrix_voxels = matrix == 1

    # Compute stats from matrix voxels only, ignoring NaNs in the data
    matrix_data = np.where(matrix_voxels, v, np.nan)

    mean_matrix = float(np.nanmean(matrix_data))
    std_matrix = float(np.nanstd(matrix_data, ddof=0))


    # Handle pathological cases
    if not np.isfinite(std_matrix) or std_matrix == 0.0:
        std_matrix = 1.0

    # Z-score everywhere, preserving NaNs
    z = (v - mean_matrix) / std_matrix
    z[~np.isfinite(v)] = np.nan

    stats = {
        "mean_matrix": mean_matrix,
        "std_matrix": std_matrix,
        "n_matrix": int(np.isfinite(matrix_data).sum()),
    }

    return z, stats
