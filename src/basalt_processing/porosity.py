from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PorosityStatistics:
    total_fraction: float
    by_z: np.ndarray
    pore_voxels: int
    mask_voxels: int


def calculate_porosity(porespace: np.ndarray, cylinder_mask: np.ndarray) -> PorosityStatistics:
    """Calculate total and per-Z porosity within a non-zero cylinder mask."""
    pores = np.asarray(porespace)
    mask = np.asarray(cylinder_mask)
    if pores.ndim != 3 or mask.ndim != 3 or pores.shape != mask.shape:
        raise ValueError("porespace and cylinder_mask must be 3-D arrays with the same shape")

    inside = mask != 0
    pore_inside = (pores != 0) & inside
    mask_by_z = inside.sum(axis=(1, 2))
    pores_by_z = pore_inside.sum(axis=(1, 2))
    by_z = np.divide(
        pores_by_z,
        mask_by_z,
        out=np.full(mask_by_z.shape, np.nan),
        where=mask_by_z != 0,
    )
    mask_voxels = int(mask_by_z.sum())
    pore_voxels = int(pores_by_z.sum())
    total_fraction = float(pore_voxels / mask_voxels) if mask_voxels else float("nan")
    return PorosityStatistics(total_fraction, by_z, pore_voxels, mask_voxels)
