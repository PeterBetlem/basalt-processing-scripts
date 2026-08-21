from __future__ import annotations

import h5py
import numpy as np
import pytest

import basalt_processing.cylinder_crop as cylinder_crop
from basalt_processing.cylinder_crop import (
    apply_cylinder_mask,
    build_cylinder_mask,
    detect_circle_on_slice,
    fit_cylinder_parameters,
    write_cylinder_crop,
)


def test_build_cylinder_mask_uses_a_constant_xy_center_for_every_z_slice() -> None:
    mask = build_cylinder_mask((3, 5, 5), radius=1.1, center_xy=np.array([2.0, 2.0]))

    expected_slice = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, np.repeat(expected_slice[None, :, :], 3, axis=0))


def test_build_cylinder_mask_follows_a_moving_centerline() -> None:
    centers_xy = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    mask = build_cylinder_mask((3, 5, 5), radius=0.5, centers_xy=centers_xy)

    np.testing.assert_array_equal(mask.sum(axis=(1, 2)), np.array([1, 1, 1]))
    assert mask[0, 1, 1] == 1
    assert mask[1, 2, 2] == 1
    assert mask[2, 3, 3] == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"radius": 0.0, "center_xy": np.array([1.0, 1.0])}, "positive"),
        ({"radius": 1.0, "center_xy": np.array([np.nan, 1.0])}, "finite"),
        ({"radius": 1.0}, "exactly one"),
        ({"radius": 1.0, "center_xy": np.array([1.0, 1.0]), "centers_xy": np.ones((2, 2))}, "exactly one"),
    ],
)
def test_build_cylinder_mask_validates_its_geometry(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_cylinder_mask((2, 3, 4), **kwargs)


def test_apply_cylinder_mask_preserves_volume_dtype() -> None:
    volume = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
    mask = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)

    masked = apply_cylinder_mask(volume, mask)

    assert masked.dtype == np.uint16
    np.testing.assert_array_equal(masked, volume * mask)


def test_fit_cylinder_parameters_fits_a_straight_centerline() -> None:
    detections = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [2.0, 12.0, 23.0, 5.0],
            [4.0, 14.0, 26.0, 5.0],
            [6.0, 16.0, 29.0, 5.0],
        ]
    )

    parameters = fit_cylinder_parameters(detections, z_size=7, radius_adjustment=0.5)

    np.testing.assert_allclose(parameters.centerline_xy, [[10, 20], [11, 21.5], [12, 23], [13, 24.5], [14, 26], [15, 27.5], [16, 29]])
    assert parameters.radius == 5.5


def test_fit_cylinder_parameters_excludes_a_single_extreme_detection() -> None:
    detections = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 11.0, 21.5, 5.0],
            [2.0, 12.0, 23.0, 5.0],
            [3.0, 13.0, 24.5, 5.0],
            [4.0, 14.0, 26.0, 5.0],
            [5.0, 500.0, 500.0, 100.0],
        ]
    )

    parameters = fit_cylinder_parameters(detections, z_size=5)

    np.testing.assert_allclose(parameters.centerline_xy, [[10, 20], [11, 21.5], [12, 23], [13, 24.5], [14, 26]])
    assert parameters.radius == 5.0


def test_fit_cylinder_parameters_rejects_three_rows_when_one_is_an_outlier() -> None:
    detections = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 11.0, 21.5, 5.0],
            [2.0, 500.0, 500.0, 100.0],
        ]
    )

    with pytest.raises(ValueError, match="at least three"):
        fit_cylinder_parameters(detections, z_size=3)


def test_fit_cylinder_parameters_keeps_a_valid_three_point_straight_centerline() -> None:
    detections = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [1.0, 2.0, 2.0, 4.0],
            [2.0, 3.0, 2.0, 4.0],
        ]
    )

    parameters = fit_cylinder_parameters(detections, z_size=3)

    np.testing.assert_allclose(parameters.centerline_xy, [[1, 2], [2, 2], [3, 2]])
    assert parameters.radius == 4.0


def test_fit_cylinder_parameters_rejects_an_outlier_in_a_constant_center_column() -> None:
    detections = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 10.0, 20.0, 5.0],
            [2.0, 500.0, 20.0, 5.0],
        ]
    )

    with pytest.raises(ValueError, match="at least three"):
        fit_cylinder_parameters(detections, z_size=3)


def test_fit_cylinder_parameters_rejects_an_extreme_z_detection() -> None:
    detections = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 10.0, 20.0, 5.0],
            [500.0, 10.0, 20.0, 5.0],
        ]
    )

    with pytest.raises(ValueError, match="at least three"):
        fit_cylinder_parameters(detections, z_size=3)


def test_detect_circle_on_slice_normalizes_and_uses_required_hough_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_hough(image: np.ndarray, method: int, **kwargs: object) -> np.ndarray:
        captured["image"] = image
        captured["method"] = method
        captured.update(kwargs)
        return np.array([[[12.5, 20.5, 100.0]]], dtype=np.float32)

    monkeypatch.setattr(cylinder_crop.cv2, "HoughCircles", fake_hough)

    result = detect_circle_on_slice(np.array([[10.0, 20.0], [30.0, 40.0]]))

    assert result == (12.5, 20.5, 100.0)
    assert captured["image"].dtype == np.uint8
    np.testing.assert_array_equal(captured["image"], [[0, 85], [170, 255]])
    assert captured["method"] == cylinder_crop.cv2.HOUGH_GRADIENT
    assert {key: captured[key] for key in ("dp", "minDist", "param1", "param2", "minRadius", "maxRadius")} == {
        "dp": 1,
        "minDist": 50,
        "param1": 100,
        "param2": 20,
        "minRadius": 100,
        "maxRadius": 800,
    }


def test_detect_circle_on_slice_requires_a_2d_image() -> None:
    with pytest.raises(ValueError, match="2-D"):
        detect_circle_on_slice(np.zeros((1, 2, 2)))


def test_write_cylinder_crop_copies_metadata_and_writes_masked_data(tmp_path) -> None:
    source = tmp_path / "input.h5"
    output = tmp_path / "output.h5"
    data = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
    cylinder_mask = np.array(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0]],
            [[1, 1, 0, 0], [0, 0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    with h5py.File(source, "w") as h5:
        h5.attrs["project"] = "basalt"
        dset = h5.create_dataset("data", data=data, chunks=(1, 2, 4))
        dset.attrs["axis_order"] = "Z,Y,X"

    result = write_cylinder_crop(source, output, cylinder_mask=cylinder_mask, slab_depth=1)

    assert result == output
    with h5py.File(output, "r") as h5:
        assert h5.attrs["project"] == "basalt"
        assert h5["data"].attrs["axis_order"] == "Z,Y,X"
        np.testing.assert_array_equal(h5["data"], data)
        np.testing.assert_array_equal(h5["cylinder_mask"], cylinder_mask)
        np.testing.assert_array_equal(h5["data_masked"], data * cylinder_mask)


def test_write_cylinder_crop_rejects_non_binary_mask(tmp_path) -> None:
    source = tmp_path / "input.h5"
    with h5py.File(source, "w") as h5:
        h5.create_dataset("data", data=np.ones((1, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="binary"):
        write_cylinder_crop(source, tmp_path / "output.h5", cylinder_mask=np.array([[[0, 1], [2, 0]]]))


def test_write_cylinder_crop_requires_a_separate_output_path(tmp_path) -> None:
    source = tmp_path / "input.h5"
    with h5py.File(source, "w") as h5:
        h5.create_dataset("data", data=np.ones((1, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="separate"):
        write_cylinder_crop(source, source, cylinder_mask=np.ones((1, 2, 2), dtype=np.uint8))


@pytest.mark.parametrize(
    ("source_contents", "cylinder_mask", "exception", "match"),
    [
        ({}, np.ones((1, 2, 2), dtype=np.uint8), KeyError, "not found"),
        ({"data": np.ones((2, 2), dtype=np.uint8)}, np.ones((2, 2), dtype=np.uint8), ValueError, "3-D"),
        ({"data": np.ones((1, 2, 2), dtype=np.uint8)}, np.ones((1, 2, 3), dtype=np.uint8), ValueError, "match"),
        ({"data": np.ones((1, 2, 2), dtype=np.uint8)}, np.array([[[0, 1], [2, 0]]]), ValueError, "binary"),
    ],
)
def test_write_cylinder_crop_preserves_existing_output_when_source_validation_fails(
    tmp_path, source_contents, cylinder_mask, exception, match
) -> None:
    source = tmp_path / "input.h5"
    output = tmp_path / "output.h5"
    expected_output = np.array([42], dtype=np.uint8)
    with h5py.File(source, "w") as h5:
        for dataset, data in source_contents.items():
            h5.create_dataset(dataset, data=data)
    with h5py.File(output, "w") as h5:
        h5.attrs["keep"] = "this output"
        h5.create_dataset("existing", data=expected_output)

    with pytest.raises(exception, match=match):
        write_cylinder_crop(source, output, cylinder_mask=cylinder_mask)

    with h5py.File(output, "r") as h5:
        assert h5.attrs["keep"] == "this output"
        np.testing.assert_array_equal(h5["existing"], expected_output)


def test_write_cylinder_crop_preserves_existing_output_when_input_is_missing(tmp_path) -> None:
    output = tmp_path / "output.h5"
    with h5py.File(output, "w") as h5:
        h5.create_dataset("existing", data=np.array([42], dtype=np.uint8))

    with pytest.raises(FileNotFoundError):
        write_cylinder_crop(tmp_path / "missing.h5", output, cylinder_mask=np.ones((1, 2, 2), dtype=np.uint8))

    with h5py.File(output, "r") as h5:
        np.testing.assert_array_equal(h5["existing"], [42])


def test_write_cylinder_crop_preserves_existing_output_when_temporary_write_fails(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.h5"
    output = tmp_path / "output.h5"
    with h5py.File(source, "w") as h5:
        h5.create_dataset("data", data=np.ones((1, 2, 2), dtype=np.uint8))
    with h5py.File(output, "w") as h5:
        h5.create_dataset("existing", data=np.array([42], dtype=np.uint8))

    def fail_to_mask(*args: object) -> np.ndarray:
        raise RuntimeError("simulated slab write failure")

    monkeypatch.setattr(cylinder_crop, "apply_cylinder_mask", fail_to_mask)

    with pytest.raises(RuntimeError, match="simulated"):
        write_cylinder_crop(source, output, cylinder_mask=np.ones((1, 2, 2), dtype=np.uint8))

    with h5py.File(output, "r") as h5:
        np.testing.assert_array_equal(h5["existing"], [42])
    assert not list(tmp_path.glob(".output.h5.*.tmp"))
