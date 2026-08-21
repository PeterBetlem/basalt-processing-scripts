from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import h5py
import numpy as np

from basalt_processing.paths import ensure_parent


@dataclass(frozen=True)
class CylinderParameters:
    centerline_xy: np.ndarray
    radius: float


def build_cylinder_mask(
    shape_zyx: tuple[int, int, int],
    *,
    radius: float,
    center_xy: np.ndarray | None = None,
    centers_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Build a uint8 Z-Y-X mask for a cylindrical region."""
    if len(shape_zyx) != 3 or any(size <= 0 for size in shape_zyx):
        raise ValueError("shape_zyx must contain three positive dimensions")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if (center_xy is None) == (centers_xy is None):
        raise ValueError("provide exactly one of center_xy or centers_xy")

    z_size, y_size, x_size = shape_zyx
    if center_xy is not None:
        center = np.asarray(center_xy, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center_xy must be one finite (X, Y) pair")
        centerline = np.repeat(center[None, :], z_size, axis=0)
    else:
        centerline = np.asarray(centers_xy, dtype=float)
        if centerline.shape != (z_size, 2) or not np.all(np.isfinite(centerline)):
            raise ValueError("centers_xy must be a finite (Z, 2) centerline")

    y, x = np.ogrid[:y_size, :x_size]
    mask = np.empty(shape_zyx, dtype=np.uint8)
    radius_squared = float(radius) ** 2
    for z_index, (center_x, center_y) in enumerate(centerline):
        mask[z_index] = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius_squared)
    return mask


def apply_cylinder_mask(volume: np.ndarray, cylinder_mask: np.ndarray) -> np.ndarray:
    """Set values outside a cylinder mask to zero without changing volume dtype."""
    volume_array = np.asarray(volume)
    mask_array = np.asarray(cylinder_mask)
    if volume_array.ndim != 3 or mask_array.ndim != 3 or volume_array.shape != mask_array.shape:
        raise ValueError("volume and cylinder_mask must be same-shaped 3-D arrays")
    return volume_array * mask_array.astype(volume_array.dtype, copy=False)


def detect_circle_on_slice(image_2d: np.ndarray) -> tuple[float, float, float] | None:
    """Detect the first circular cross-section in one image slice."""
    image = np.asarray(image_2d)
    if image.ndim != 2:
        raise ValueError("image_2d must be a 2-D array")
    minimum = float(np.nanmin(image))
    maximum = float(np.nanmax(image))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("image_2d must contain finite values")
    if maximum == minimum:
        normalized = np.zeros(image.shape, dtype=np.uint8)
    else:
        normalized = np.clip((image - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        normalized,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=20,
        minRadius=100,
        maxRadius=800,
    )
    if circles is None:
        return None
    x, y, radius = circles[0, 0]
    return float(x), float(y), float(radius)


def fit_cylinder_parameters(
    detections_zxyr: np.ndarray,
    *,
    z_size: int,
    radius_adjustment: float = 0.0,
) -> CylinderParameters:
    """Fit a straight cylinder centerline from Z-X-Y-radius detections."""
    detections = np.asarray(detections_zxyr, dtype=float)
    if detections.ndim != 2 or detections.shape[1] != 4:
        raise ValueError("detections_zxyr must have shape (N, 4)")
    if z_size <= 0:
        raise ValueError("z_size must be positive")
    finite = detections[np.all(np.isfinite(detections), axis=1)]
    if len(finite) >= 3:
        features = finite
        median = np.median(features, axis=0)
        median_absolute_deviation = np.median(np.abs(features - median), axis=0)
        nonzero_deviation = median_absolute_deviation > 0
        modified_z_scores = np.zeros_like(features)
        modified_z_scores[:, nonzero_deviation] = (
            0.6745
            * np.abs(features[:, nonzero_deviation] - median[nonzero_deviation])
            / median_absolute_deviation[nonzero_deviation]
        )
        zero_deviation_outliers = (~nonzero_deviation) & ~np.isclose(
            features,
            median,
            rtol=1e-9,
            atol=1e-12,
        )
        modified_z_scores[zero_deviation_outliers] = np.inf
        finite = finite[np.all(modified_z_scores < 3.5, axis=1)]
    if len(finite) < 3:
        raise ValueError("at least three finite, non-outlier detections are required")

    z, x, y, radii = finite.T
    x_coefficients = np.polyfit(z, x, deg=1)
    y_coefficients = np.polyfit(z, y, deg=1)
    all_z = np.arange(z_size, dtype=float)
    centerline = np.column_stack((np.polyval(x_coefficients, all_z), np.polyval(y_coefficients, all_z)))
    radius = float(np.mean(radii) + radius_adjustment)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("final radius must be finite and positive")
    return CylinderParameters(centerline_xy=centerline, radius=radius)


def _dataset_chunks(shape: tuple[int, int, int], source_chunks: tuple[int, ...] | None) -> tuple[int, int, int]:
    if source_chunks is not None and len(source_chunks) == 3:
        return tuple(min(size, chunk) for size, chunk in zip(shape, source_chunks))
    return tuple(min(size, 64) for size in shape)


def write_cylinder_crop(
    input_h5: str | Path,
    output_h5: str | Path,
    *,
    cylinder_mask: np.ndarray,
    data_dataset: str = "data",
    mask_dataset: str = "cylinder_mask",
    masked_dataset: str = "data_masked",
    slab_depth: int = 64,
) -> Path:
    """Write copied data, mask, and masked data to a separate HDF5 output."""
    if slab_depth <= 0:
        raise ValueError("slab_depth must be positive")
    input_path = Path(input_h5)
    output_path = Path(output_h5)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_h5 and output_h5 must be separate files")
    mask_values = np.asarray(cylinder_mask)
    if not np.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("cylinder_mask must contain only binary 0/1 values")
    mask = mask_values.astype(np.uint8, copy=False)
    temp_path: Path | None = None
    with h5py.File(input_path, "r") as source:
        if data_dataset not in source:
            raise KeyError(f"Dataset {data_dataset!r} not found")
        source_data = source[data_dataset]
        if not isinstance(source_data, h5py.Dataset):
            raise ValueError(f"Dataset {data_dataset!r} must be an HDF5 dataset")
        if source_data.ndim != 3 or mask.shape != source_data.shape:
            raise ValueError("cylinder_mask must match the source data's 3-D shape")
        output_path = ensure_parent(output_path)
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temp_path = Path(temporary_file.name)
        try:
            with h5py.File(temp_path, "w") as output:
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                chunks = _dataset_chunks(source_data.shape, source_data.chunks)
                temp_data = output.create_dataset("__data_tmp", shape=source_data.shape, dtype=source_data.dtype, chunks=chunks, compression="gzip")
                temp_mask = output.create_dataset("__mask_tmp", data=mask, chunks=chunks, compression="gzip")
                temp_masked = output.create_dataset("__masked_tmp", shape=source_data.shape, dtype=source_data.dtype, chunks=chunks, compression="gzip")
                for key, value in source_data.attrs.items():
                    temp_data.attrs[key] = value
                for start in range(0, source_data.shape[0], slab_depth):
                    stop = min(start + slab_depth, source_data.shape[0])
                    slab = source_data[start:stop]
                    temp_data[start:stop] = slab
                    temp_masked[start:stop] = apply_cylinder_mask(slab, mask[start:stop])
                output.move("__data_tmp", data_dataset)
                output.move("__mask_tmp", mask_dataset)
                output.move("__masked_tmp", masked_dataset)
            temp_path.replace(output_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    return output_path
