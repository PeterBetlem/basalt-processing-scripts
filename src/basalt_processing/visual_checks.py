"""Reusable plots for inspecting registered CT volumes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _as_volume(volume: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (Z, Y, X), got {array.shape}")
    if not all(array.shape):
        raise ValueError(f"{name} cannot contain an empty axis")
    return array


def _as_voxel_size(voxel_size_zyx: Sequence[float]) -> tuple[float, float, float]:
    if len(voxel_size_zyx) != 3 or any(value <= 0 for value in voxel_size_zyx):
        raise ValueError("voxel_size_zyx must contain three positive (Z, Y, X) values")
    return tuple(float(value) for value in voxel_size_zyx)


def _draw_x_slice_box(ax, x: int, ylim: tuple[int, int], zlim: tuple[int, int]) -> None:
    ymin, ymax = ylim
    zmin, zmax = zlim
    ax.plot([x, x], [ymin, ymax], [zmin, zmin], color="m", lw=1.5, alpha=0.8)
    ax.plot(
        [x, x], [ymin, ymax], [zmax, zmax], color="m", lw=1.5, alpha=0.8, zorder=1000
    )
    ax.plot(
        [x, x], [ymin, ymin], [zmin, zmax], color="m", lw=1.5, alpha=0.8, zorder=1000
    )
    ax.plot([x, x], [ymax, ymax], [zmin, zmax], color="m", lw=1.5, alpha=0.8)


def _draw_y_slice_box(ax, y: int, xlim: tuple[int, int], zlim: tuple[int, int]) -> None:
    xmin, xmax = xlim
    zmin, zmax = zlim
    ax.plot([xmin, xmax], [y, y], [zmin, zmin], color="c", lw=1.5, alpha=0.8)
    ax.plot(
        [xmin, xmax], [y, y], [zmax, zmax], color="c", lw=1.5, alpha=0.8, zorder=1000
    )
    ax.plot(
        [xmin, xmin], [y, y], [zmin, zmax], color="c", lw=1.5, alpha=0.8, zorder=1000
    )
    ax.plot([xmax, xmax], [y, y], [zmin, zmax], color="c", lw=1.5, alpha=0.8)


def _set_mm_ticklabels(
    ax,
    voxel_size_zyx: Sequence[float],
    *,
    stride: int = 1,
    set_zero: bool = True,
) -> None:
    dz, dy, dx = voxel_size_zyx
    zero_x = ax.get_xlim()[0] * dx if set_zero else 0.0
    zero_y = ax.get_ylim()[0] * dy if set_zero else 0.0
    zero_z = 0.0
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(np.round(value * dx - zero_x))}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(np.round(value * dy - zero_y))}")
    )
    ax.zaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: f"{int(np.round((value * dz - zero_z) / stride))}"
        )
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")


def _style_orthogonal_row_axis(
    ax,
    *,
    spine_color: str,
    x_position: str,
) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)

    ax.xaxis.set_ticks_position(x_position)
    ax.xaxis.set_label_position(x_position)
    ax.tick_params(
        axis="x",
        top=x_position == "top",
        labeltop=x_position == "top",
        bottom=x_position == "bottom",
        labelbottom=x_position == "bottom",
        colors="black",
        labelcolor="black",
    )
    ax.tick_params(axis="y", colors="black", labelcolor="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def contourf_slices_3d(
    volume: np.ndarray,
    z_indices: Iterable[int],
    levels: int = 30,
    stride: int = 2,
    alpha: float = 0.8,
    cmap: str = "viridis",
    outline_color: str = "k",
    outline_lw: float = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    voxel_size_zyx: Sequence[float] = (0.03125, 0.03125, 0.03125),
    voxel_size: Sequence[float] | None = None,
    draw_x_box: bool = False,
    draw_y_box: bool = False,
    x_index: int | None = None,
    y_index: int | None = None,
    ax=None,
):
    """Render selected Z planes from a ``(Z, Y, X)`` volume on a 3D axis."""
    if stride < 1:
        raise ValueError("stride must be at least one")
    if levels < 2:
        raise ValueError("levels must be at least two")

    array = _as_volume(volume, "volume")
    # ``voxel_size`` is the name used by the source notebook. Keep the newer
    # explicit ``voxel_size_zyx`` spelling too, because it documents the axis
    # order used by the registered volumes.
    dz, dy, dx = _as_voxel_size(voxel_size_zyx if voxel_size is None else voxel_size)
    sampled = array[:, ::stride, ::stride]
    z_size, y_size, x_size = sampled.shape
    finite = np.isfinite(sampled)

    if vmin is None:
        vmin = float(np.nanmin(sampled[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(sampled[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        margin = 0.5 if vmin == 0 else abs(vmin) * 0.01
        vmin -= margin
        vmax += margin

    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    if ax is None:
        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot(projection="3d")

    for z_index in z_indices:
        if not 0 <= z_index < z_size:
            continue
        plane = sampled[z_index]
        ax.contourf(
            x_grid,
            y_grid,
            np.ma.masked_invalid(plane),
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zdir="z",
            offset=z_index,
            alpha=alpha,
        )
        support = np.isfinite(plane).astype(float)
        if support.min() < 0.5 < support.max():
            ax.contour(
                x_grid,
                y_grid,
                support,
                levels=[0.5],
                colors=[outline_color],
                linewidths=outline_lw,
                zdir="z",
                offset=z_index,
            )

    # Keep the source figure's coordinate extents and Z orientation.  In
    # particular, setting the reversed Z limit followed by invert_zaxis is
    # intentional: it matches the source notebook's displayed convention.
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_zlim(z_size, 0)
    ax.invert_zaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(axis="x", pad=-6)
    ax.tick_params(axis="y", pad=0)
    ax.tick_params(axis="z", pad=-6)
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.zaxis.labelpad = -12
    ax.set_xticks(np.arange(0, x_size, step=10 / dx / stride))
    ax.set_yticks(np.arange(0, y_size, step=10 / dy / stride))
    ax.set_zticks(np.arange(0, z_size, step=10 / dz))
    _set_mm_ticklabels(
        ax,
        tuple(value * stride for value in (dz, dy, dx)),
        stride=stride,
    )
    ax.set_box_aspect((1, 1, 1))

    if draw_x_box:
        source_x = 500 if x_index is None else x_index
        _draw_x_slice_box(
            ax,
            int(np.clip(source_x // stride, 0, x_size)),
            (0, y_size),
            (0, z_size),
        )
    if draw_y_box:
        source_y = 550 if y_index is None else y_index
        _draw_y_slice_box(
            ax,
            int(np.clip(source_y // stride, 0, y_size)),
            (0, x_size),
            (0, z_size),
        )

    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.zaxis.set_rotate_label(270)
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=-15,
        verticalalignment="baseline",
        horizontalalignment="left",
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=40,
        verticalalignment="top",
        horizontalalignment="right",
    )
    ax.set_zticklabels(
        ax.get_zticklabels(),
        rotation=40,
        verticalalignment="top",
        horizontalalignment="right",
    )

    return ax


def plot_combined_orthogonal_comparison(
    reference: np.ndarray,
    differences: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    z_indices: Iterable[int],
    x_index: int,
    y_index: int,
    voxel_size_zyx: Sequence[float],
    reference_limits: tuple[float | None, float | None] = (None, None),
    difference_limits: tuple[float | None, float | None] = (-5.0, 5.0),
    figure_size: tuple[float, float] = (12.0, 8.0),
    dpi: int = 200,
):
    """Plot reference and up to three normalized differences in 2D and 3D."""
    reference_array = _as_volume(reference, "reference")
    difference_arrays = [
        _as_volume(difference, "difference") for difference in differences
    ]
    if len(difference_arrays) > 3:
        raise ValueError("at most three difference volumes can be shown")
    if len(labels) != len(difference_arrays) + 1:
        raise ValueError(
            "labels must contain one reference label followed by one per difference"
        )
    if any(
        difference.shape != reference_array.shape for difference in difference_arrays
    ):
        raise ValueError("all difference volumes must have the reference shape")

    dz, dy, dx = _as_voxel_size(voxel_size_zyx)
    z_size, y_size, x_size = reference_array.shape
    x_index = int(np.clip(x_index, 0, x_size - 1))
    y_index = int(np.clip(y_index, 0, y_size - 1))
    y_extent = (0, y_size * dy, 0, z_size * dz)
    x_extent = (0, x_size * dx, 0, z_size * dz)

    figure = plt.figure(figsize=figure_size, dpi=dpi)
    axes = np.empty((3, 4), dtype=object)
    for row in range(3):
        for column in range(4):
            axes[row, column] = (
                figure.add_subplot(3, 4, row * 4 + column + 1, projection="3d")
                if row == 1
                else figure.add_subplot(3, 4, row * 4 + column + 1)
            )

    ref_vmin, ref_vmax = reference_limits
    diff_vmin, diff_vmax = difference_limits
    axes[0, 0].imshow(
        reference_array[:, :, x_index],
        cmap="gray",
        extent=y_extent,
        origin="lower",
        aspect="equal",
        vmin=ref_vmin,
        vmax=ref_vmax,
    )
    axes[2, 0].imshow(
        reference_array[:, y_index, :],
        cmap="gray",
        extent=x_extent,
        origin="lower",
        aspect="equal",
        vmin=ref_vmin,
        vmax=ref_vmax,
    )
    axes[0, 0].set_title(labels[0])

    for column, (difference, label) in enumerate(
        zip(difference_arrays, labels[1:]), start=1
    ):
        axes[0, column].imshow(
            difference[:, :, x_index],
            cmap="bwr",
            extent=y_extent,
            origin="lower",
            aspect="equal",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        axes[2, column].imshow(
            difference[:, y_index, :],
            cmap="bwr",
            extent=x_extent,
            origin="lower",
            aspect="equal",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        axes[0, column].set_title(label)

    contourf_slices_3d(
        reference_array,
        z_indices,
        cmap="gray",
        alpha=0.4,
        vmin=ref_vmin,
        vmax=ref_vmax,
        voxel_size_zyx=(dz, dy, dx),
        draw_x_box=True,
        draw_y_box=True,
        x_index=x_index,
        y_index=y_index,
        ax=axes[1, 0],
    )
    for column, difference in enumerate(difference_arrays, start=1):
        contourf_slices_3d(
            difference,
            z_indices,
            cmap="bwr",
            alpha=0.5,
            vmin=diff_vmin,
            vmax=diff_vmax,
            voxel_size_zyx=(dz, dy, dx),
            draw_x_box=True,
            draw_y_box=True,
            x_index=x_index,
            y_index=y_index,
            ax=axes[1, column],
        )

    for column in range(4):
        for row in (0, 2):
            axes[row, column].set_xlabel("Y (mm)" if row == 0 else "X (mm)")
            axes[row, column].set_ylabel("Z (mm)")
        if column > len(difference_arrays):
            axes[0, column].axis("off")
            axes[1, column].set_axis_off()
            axes[2, column].axis("off")
        else:
            _style_orthogonal_row_axis(
                axes[0, column],
                spine_color="m",
                x_position="top",
            )
            _style_orthogonal_row_axis(
                axes[2, column],
                spine_color="c",
                x_position="bottom",
            )

    figure.subplots_adjust(wspace=0.6, hspace=0.7)
    for axis in axes[1]:
        position = axis.get_position()
        pad_w, pad_h = 0.04, 0.04
        axis.set_position(
            [
                position.x0 - pad_w,
                position.y0 - pad_h,
                position.width + 2 * pad_w,
                position.height + 2.8 * pad_h,
            ]
        )
        axis.zaxis.set_label_position("upper")
    return figure, axes
