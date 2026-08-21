import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from basalt_processing.visual_checks import (
    contourf_slices_3d,
    plot_combined_orthogonal_comparison,
)


def test_contourf_slices_3d_draws_selected_slices_on_provided_axis():
    volume = np.arange(4 * 6 * 8, dtype=float).reshape(4, 6, 8)
    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")

    result = contourf_slices_3d(
        volume,
        z_indices=[0, 3],
        levels=4,
        stride=2,
        voxel_size_zyx=(0.5, 0.25, 0.25),
        ax=axis,
    )

    assert result is axis
    assert axis.get_xlabel() == "X (mm)"
    assert axis.get_ylabel() == "Y (mm)"
    assert axis.get_zlabel() == "Z (mm)"
    assert axis.collections
    assert axis.get_xlim() == (0.0, 4.0)
    assert axis.get_ylim() == (0.0, 3.0)
    assert axis.get_zlim() == (0.0, 4.0)
    np.testing.assert_allclose(
        axis.get_box_aspect(),
        np.full(3, axis.get_box_aspect()[0]),
    )
    plt.close(figure)


def test_contourf_slices_3d_accepts_source_voxel_size_argument():
    volume = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")

    contourf_slices_3d(
        volume,
        [0],
        levels=4,
        voxel_size=(0.5, 0.25, 0.25),
        ax=axis,
    )

    assert axis.get_xlabel() == "X (mm)"
    plt.close(figure)


def test_combined_orthogonal_comparison_builds_2d_and_3d_panels():
    reference = np.arange(4 * 6 * 8, dtype=float).reshape(4, 6, 8)
    differences = [np.full_like(reference, fill_value) for fill_value in (1.0, 2.0, 3.0)]

    figure, axes = plot_combined_orthogonal_comparison(
        reference,
        differences,
        labels=["reference", "first", "second", "third"],
        z_indices=[0, 3],
        x_index=2,
        y_index=3,
        voxel_size_zyx=(0.5, 0.25, 0.25),
    )

    assert axes.shape == (3, 4)
    assert axes[1, 0].name == "3d"
    assert axes[1, 0].get_position().width > axes[0, 0].get_position().width
    assert axes[1, 0].get_position().height > axes[0, 0].get_position().height
    assert len(axes[0, 0].images) == 1
    assert len(axes[2, 3].images) == 1
    plt.close(figure)


def test_combined_orthogonal_comparison_styles_orthogonal_rows():
    reference = np.arange(4 * 6 * 8, dtype=float).reshape(4, 6, 8)
    differences = [np.full_like(reference, fill_value) for fill_value in (1.0, 2.0)]

    figure, axes = plot_combined_orthogonal_comparison(
        reference,
        differences,
        labels=["reference", "first", "second"],
        z_indices=[0, 3],
        x_index=2,
        y_index=3,
        voxel_size_zyx=(0.5, 0.25, 0.25),
    )
    figure.canvas.draw()

    top_axis = axes[0, 0]
    bottom_axis = axes[2, 0]

    assert top_axis.xaxis.get_ticks_position() == "top"
    assert top_axis.xaxis.get_label_position() == "top"
    assert bottom_axis.xaxis.get_ticks_position() == "bottom"
    assert bottom_axis.xaxis.get_label_position() == "bottom"

    for spine in top_axis.spines.values():
        assert spine.get_edgecolor() == to_rgba("m")
    for spine in bottom_axis.spines.values():
        assert spine.get_edgecolor() == to_rgba("c")

    for axis in (top_axis, bottom_axis):
        assert axis.xaxis.label.get_color() == "black"
        assert axis.yaxis.label.get_color() == "black"
        assert all(label.get_color() == "black" for label in axis.get_xticklabels())
        assert all(label.get_color() == "black" for label in axis.get_yticklabels())

    assert top_axis.xaxis.get_major_ticks()[0].tick2line.get_color() == "black"
    assert bottom_axis.xaxis.get_major_ticks()[0].tick1line.get_color() == "black"
    plt.close(figure)
