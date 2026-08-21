import numpy as np
import SimpleITK as sitk

from basalt_processing.registration import (
    crop_zero_planes,
    mask_physical_bounds,
    make_global_reference_slab,
    shrink_for_registration,
)


def test_crop_zero_planes_finds_bounds_without_a_full_volume_mask():
    volume = np.zeros((7, 8, 9), dtype=np.float32)
    volume[2:6, 3:7, 1:5] = 2.0

    cropped, slices, offset = crop_zero_planes(volume)

    assert cropped.shape == (4, 4, 4)
    assert slices == (slice(2, 6), slice(3, 7), slice(1, 5))
    assert offset == (2, 3, 1)
    np.testing.assert_array_equal(cropped, volume[slices])


def test_shrink_for_registration_limits_voxel_count():
    image = sitk.Image(100, 100, 100, sitk.sitkFloat32)
    image.SetSpacing((0.1, 0.2, 0.3))

    reduced, factors = shrink_for_registration(image, max_voxels=16_000)

    assert factors == (4, 4, 4)
    assert reduced.GetSize() == (25, 25, 25)
    assert reduced.GetSpacing() == (0.4, 0.8, 1.2)


def test_shrink_for_registration_preserves_legacy_full_resolution_when_disabled():
    image = sitk.Image(100, 100, 100, sitk.sitkFloat32)

    registration_image, factors = shrink_for_registration(image, max_voxels=None)

    assert registration_image is image
    assert factors == (1, 1, 1)


def test_global_reference_slab_has_matching_geometry():
    slab = make_global_reference_slab(
        origin_xyz=(10.0, 20.0, 30.0),
        spacing_xyz=(0.5, 1.5, 2.0),
        size_xyz=(6, 5, 8),
        z_start=3,
        z_stop=6,
    )

    assert slab.GetSize() == (6, 5, 3)
    assert slab.GetSpacing() == (0.5, 1.5, 2.0)
    assert slab.GetOrigin() == (10.0, 20.0, 36.0)


def test_mask_physical_bounds_returns_none_for_empty_mask():
    mask = sitk.Image(4, 3, 2, sitk.sitkUInt8)

    assert mask_physical_bounds(mask) is None


def test_mask_physical_bounds_respects_geometry_and_direction():
    array = np.zeros((4, 5, 6), dtype=np.uint8)
    array[1:4, 2:4, 1:5] = 255
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((0.5, 1.5, 2.0))
    mask.SetOrigin((10.0, 20.0, 30.0))
    mask.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    bounds = mask_physical_bounds(mask)

    assert bounds is not None
    min_xyz, max_xyz = bounds
    corners = np.array(
        [
            mask.TransformIndexToPhysicalPoint((x, y, z))
            for x in (1, 4)
            for y in (2, 3)
            for z in (1, 3)
        ]
    )
    np.testing.assert_allclose(min_xyz, corners.min(axis=0))
    np.testing.assert_allclose(max_xyz, corners.max(axis=0))
