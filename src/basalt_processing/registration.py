import time
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk

# ============================================================
# User-friendly logging / status prints
# ============================================================


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def log_step(prefix: str, i: int, n: int, path: str, extra: str = "") -> None:
    name = Path(path).name
    suffix = f" — {extra}" if extra else ""
    log(f"{prefix} ({i + 1}/{n}): {name}{suffix}")


def _fmt_tuple(t, nd=4):
    return tuple(float(f"{x:.{nd}f}") for x in t)


# -------------------------
# HDF5 loading + metadata
# -------------------------
def load_h5_volume(path, vol_key="data_masked", ref_key="data"):
    with h5py.File(path, "r") as f:
        dset = f[vol_key]
        # Convert while HDF5 reads so a full-size source-dtype array is not
        # allocated before the float32 result.
        vol = dset.astype(np.float32)[...]

        axis_order = f[ref_key].attrs.get("axis_order", "Z,Y,X")
        if axis_order != "Z,Y,X":
            raise ValueError(f"Expected axis_order 'Z,Y,X', got '{axis_order}'")

        dz, dy, dx = f[ref_key].attrs["voxel_size"]  # (dz,dy,dx)
        spacing_xyz = (float(dx), float(dy), float(dz))  # SITK expects (x,y,z)

        unit = f[ref_key].attrs.get("unit", "mm")
        if unit != "mm":
            raise ValueError(f"Expected unit 'mm', got '{unit}'")

    return vol, spacing_xyz


# ---------------------------------------
# Crop zero-only planes along Z/Y/X
# ---------------------------------------
def crop_zero_planes(vol):
    """
    Crops planes that are entirely zero along Z/Y/X.
    Returns:
      cropped_vol
      slices_zyx  : (slice(z0,z1), slice(y0,y1), slice(x0,x1))
      offset_zyx  : (z0,y0,x0)  start index in the original array
    """
    if vol.ndim != 3:
        raise ValueError(f"vol must be 3D (Z,Y,X), got shape {vol.shape}")

    # Scan in bounded-size Z slabs.  Creating ``vol != 0`` for the entire
    # volume adds one byte per voxel to peak memory (about 1 GiB for a
    # billion-voxel scan), while only three 1D projections are needed here.
    z_any = np.zeros(vol.shape[0], dtype=bool)
    y_any = np.zeros(vol.shape[1], dtype=bool)
    x_any = np.zeros(vol.shape[2], dtype=bool)
    voxels_per_plane = max(1, vol.shape[1] * vol.shape[2])
    target_mask_bytes = 64 * 1024**2
    slab_depth = max(1, target_mask_bytes // voxels_per_plane)

    for z_start in range(0, vol.shape[0], slab_depth):
        z_stop = min(z_start + slab_depth, vol.shape[0])
        slab_mask = vol[z_start:z_stop] != 0
        z_any[z_start:z_stop] = slab_mask.any(axis=(1, 2))
        y_any |= slab_mask.any(axis=(0, 2))
        x_any |= slab_mask.any(axis=(0, 1))

    if not z_any.any():
        # all zero; return minimal crop to avoid errors
        return vol[:1, :1, :1], (slice(0, 1), slice(0, 1), slice(0, 1)), (0, 0, 0)

    z = np.where(z_any)[0]
    y = np.where(y_any)[0]
    x = np.where(x_any)[0]

    z0, z1 = int(z[0]), int(z[-1] + 1)
    y0, y1 = int(y[0]), int(y[-1] + 1)
    x0, x1 = int(x[0]), int(x[-1] + 1)

    cropped = vol[z0:z1, y0:y1, x0:x1]
    slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    offset = (z0, y0, x0)
    return cropped, slices, offset


def make_mask_and_filled(vol):
    m = (vol != 0).astype(np.uint8)
    # np.where already returns float32 when ``vol`` is float32.  A following
    # astype() with its default copy=True would duplicate the whole volume.
    filled = np.where(m, vol, np.float32(0.0)).astype(np.float32, copy=False)
    return m, filled


def to_sitk(arr_zyx, spacing_xyz, origin_xyz=(0.0, 0.0, 0.0), is_mask=False):
    # Avoid copying an already correctly typed NumPy array before SimpleITK
    # performs its own image allocation.
    arr_zyx = np.asarray(arr_zyx, dtype=np.uint8 if is_mask else np.float32)
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(tuple(map(float, spacing_xyz)))  # (dx,dy,dz)
    img.SetOrigin(tuple(map(float, origin_xyz)))  # (ox,oy,oz) in mm
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def crop_offset_to_origin_xyz_mm(offset_zyx, spacing_xyz):
    z0, y0, x0 = offset_zyx
    dx, dy, dz = spacing_xyz
    return (x0 * dx, y0 * dy, z0 * dz)


# -------------------------
# Registration stages
# -------------------------
def _configure_common(
    reg: sitk.ImageRegistrationMethod,
    *,
    learning_rate=2.0,
    iters=300,
    image_size=None,
):
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=123)
    reg.SetInterpolator(sitk.sitkLinear)

    # A downsampled registration image can be too small for the original
    # [4, 2, 1] pyramid; choose a valid, shallower pyramid in that case.
    min_axis_size = min(image_size) if image_size is not None else 64
    if min_axis_size >= 64:
        shrink_factors = [4, 2, 1]
    elif min_axis_size >= 32:
        shrink_factors = [2, 1]
    else:
        shrink_factors = [1]
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(list(range(len(shrink_factors) - 1, -1, -1)))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1e-4,
        numberOfIterations=int(iters),
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()


def shrink_for_registration(img: sitk.Image, max_voxels: int | None):
    """Return a registration image and its shrink factors.

    ``None`` preserves the full-resolution registration used by the legacy
    global-volume generation.  A positive limit opts into reduced-resolution
    registration for memory-constrained runs.  Registration transforms are
    expressed in physical coordinates in either case.
    """
    if max_voxels is None:
        return img, (1, 1, 1)
    if max_voxels < 1:
        raise ValueError("registration_max_voxels must be at least 1")

    size = np.asarray(img.GetSize(), dtype=np.int64)
    voxel_count = int(np.prod(size))
    factor = max(1, int(np.ceil((voxel_count / max_voxels) ** (1 / 3))))
    factors = tuple(max(1, min(factor, int(axis_size))) for axis_size in size)
    if factors == (1, 1, 1):
        return img, factors
    return sitk.Shrink(img, factors), factors


def register_rigid(fixed_img, moving_img, fixed_mask, moving_mask):
    log("  Rigid registration: starting")
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)
    _configure_common(
        reg, learning_rate=2.0, iters=300, image_size=fixed_img.GetSize()
    )

    init = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(init, inPlace=False)

    t0 = time.time()
    T_rigid = reg.Execute(fixed_img, moving_img)
    log(
        f"  Rigid registration: done in {time.time() - t0:.1f}s; metric={reg.GetMetricValue():.6g}"
    )
    return T_rigid  # may be CompositeTransform; that's OK


def register_affine_refine(
    fixed_img, moving_img, fixed_mask, moving_mask, initial_rigid
):
    """
    Robust affine refinement that works even if initial_rigid is a CompositeTransform.
    We apply initial_rigid as a fixed pre-transform, then optimize an affine on top,
    and return the combined transform as a CompositeTransform.
    """
    log("  Affine refinement: starting")
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)

    _configure_common(
        reg, learning_rate=1.0, iters=300, image_size=fixed_img.GetSize()
    )

    reg.SetMovingInitialTransform(initial_rigid)

    affine = sitk.AffineTransform(3)
    reg.SetInitialTransform(affine, inPlace=False)

    t0 = time.time()
    T_aff = reg.Execute(fixed_img, moving_img)
    log(
        f"  Affine refinement: done in {time.time() - t0:.1f}s; metric={reg.GetMetricValue():.6g}"
    )

    total = sitk.CompositeTransform(3)
    total.AddTransform(initial_rigid)
    total.AddTransform(T_aff)
    return total


# -------------------------
# Resampling
# -------------------------
def resample(moving_img, reference_img, transform, is_mask=False, default_value=0.0):
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(reference_img)
    res.SetTransform(transform)
    res.SetDefaultPixelValue(float(default_value))
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    pixel_type = (
        sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    )  # NOTE: sitkFloat32 (not sitkfloat32)
    moving_cast = sitk.Cast(moving_img, pixel_type)

    out = res.Execute(moving_cast)
    return out


def mask_physical_bounds(img_mask: sitk.Image):
    """
    Returns (min_xyz, max_xyz) of mask non-zero voxels in physical coords.
    Assumes img_mask is a SimpleITK image with correct spacing/origin/direction.

    Use SimpleITK's label statistics instead of ``np.argwhere``.  The latter
    allocates one three-element int64 coordinate for every foreground voxel
    (24 bytes per voxel), which is prohibitively expensive for large or dense
    masks.
    """
    if img_mask.GetPixelID() != sitk.sitkUInt8:
        raise TypeError("img_mask must be a UInt8 SimpleITK image")

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(img_mask)
    labels = stats.GetLabels()  # zero is the filter's background label
    if not labels:
        return None

    # BoundingBox is (x_min, y_min, z_min, size_x, size_y, size_z).
    boxes = np.array([stats.GetBoundingBox(label) for label in labels], dtype=np.int64)
    mins = boxes[:, :3].min(axis=0)
    maxs = (boxes[:, :3] + boxes[:, 3:] - 1).max(axis=0)
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxs

    corners_zyx = [
        (zmin, ymin, xmin),
        (zmin, ymin, xmax),
        (zmin, ymax, xmin),
        (zmin, ymax, xmax),
        (zmax, ymin, xmin),
        (zmax, ymin, xmax),
        (zmax, ymax, xmin),
        (zmax, ymax, xmax),
    ]
    corners_xyz = [
        img_mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        for (z, y, x) in corners_zyx
    ]

    corners_xyz = np.array(corners_xyz, dtype=np.float64)
    min_xyz = corners_xyz.min(axis=0)
    max_xyz = corners_xyz.max(axis=0)
    return min_xyz, max_xyz


def transform_bounds(bounds_min_xyz, bounds_max_xyz, transform: sitk.Transform):
    """
    Transform 8 corners of an AABB, return transformed min/max in fixed space.
    """
    mins = np.array(bounds_min_xyz, dtype=np.float64)
    maxs = np.array(bounds_max_xyz, dtype=np.float64)

    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )

    tc = np.array(
        [transform.TransformPoint(tuple(p)) for p in corners], dtype=np.float64
    )
    return tc.min(axis=0), tc.max(axis=0)


def make_global_reference_image(origin_xyz, spacing_xyz, size_xyz):
    """
    Create an empty reference image defining the global grid.
    size_xyz: (sx,sy,sz) in voxels
    """
    img = sitk.Image(
        int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2]), sitk.sitkFloat32
    )
    img.SetOrigin(tuple(map(float, origin_xyz)))
    img.SetSpacing(tuple(map(float, spacing_xyz)))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def make_global_reference_slab(
    origin_xyz, spacing_xyz, size_xyz, z_start: int, z_stop: int
):
    """Create a Z-slab of a global reference grid without allocating all of it."""
    sx, sy, sz = map(int, size_xyz)
    z_start, z_stop = int(z_start), int(z_stop)
    if not (0 <= z_start < z_stop <= sz):
        raise ValueError(f"Invalid Z slab [{z_start}, {z_stop}) for depth {sz}")

    ox, oy, oz = map(float, origin_xyz)
    dx, dy, dz = map(float, spacing_xyz)
    return make_global_reference_image(
        (ox, oy, oz + z_start * dz), (dx, dy, dz), (sx, sy, z_stop - z_start)
    )


# ============================================================
# Transform storage in HDF5: /transforms/...
# ============================================================


def _write_single_transform(grp, T: sitk.Transform):
    grp.attrs["type"] = T.GetName()
    grp.attrs["dimension"] = int(T.GetDimension())
    grp.attrs["parameters"] = np.array(T.GetParameters(), dtype=np.float64)
    grp.attrs["fixed_parameters"] = np.array(T.GetFixedParameters(), dtype=np.float64)


def write_transform_h5(h5file: h5py.File, path: str, T: sitk.Transform):
    """
    Writes a SimpleITK transform (including CompositeTransform) to HDF5.
    """
    if path in h5file:
        del h5file[path]
    grp = h5file.create_group(path)

    if isinstance(T, sitk.CompositeTransform):
        grp.attrs["type"] = "CompositeTransform"
        grp.attrs["dimension"] = int(T.GetDimension())
        for i in range(T.GetNumberOfTransforms()):
            sub = grp.create_group(f"t{i}")
            _write_single_transform(sub, T.GetNthTransform(i))
    else:
        _write_single_transform(grp, T)


def _make_transform_from_name(ttype: str, dim: int) -> sitk.Transform:
    ttype = str(ttype)

    # ✅ Identity / generic base transform name (common for sitkIdentity)
    if ttype in ("Transform", "IdentityTransform"):
        return sitk.Transform(dim, sitk.sitkIdentity)

    # 3D rigid-ish
    if ttype == "Euler3DTransform":
        return sitk.Euler3DTransform()
    if ttype == "VersorRigid3DTransform":
        return sitk.VersorRigid3DTransform()
    if ttype == "Similarity3DTransform":
        return sitk.Similarity3DTransform()
    if ttype == "ScaleVersor3DTransform":
        return sitk.ScaleVersor3DTransform()

    # Generic N-D / common linear
    if ttype == "AffineTransform":
        return sitk.AffineTransform(dim)
    if ttype == "TranslationTransform":
        return sitk.TranslationTransform(dim)
    if ttype == "ScaleTransform":
        return sitk.ScaleTransform(dim)

    raise ValueError(
        f"Unsupported transform type '{ttype}'. Add it to _make_transform_from_name()."
    )


def _read_transform_recursive(grp) -> sitk.Transform:
    """
    Recursively reconstruct a transform (supports nested CompositeTransform).
    """
    ttype = str(grp.attrs["type"])

    if ttype == "CompositeTransform":
        dim = int(grp.attrs["dimension"])
        T = sitk.CompositeTransform(dim)

        i = 0
        while f"t{i}" in grp:
            T.AddTransform(_read_transform_recursive(grp[f"t{i}"]))
            i += 1

        return T

    # Leaf transform
    dim = int(grp.attrs["dimension"])
    params = np.array(grp.attrs["parameters"], dtype=np.float64).tolist()
    fixed = np.array(grp.attrs["fixed_parameters"], dtype=np.float64).tolist()

    T = _make_transform_from_name(ttype, dim)

    # Order matters: fixed first
    T.SetFixedParameters(tuple(fixed))
    T.SetParameters(tuple(params))
    return T


def read_transform_h5(h5file: h5py.File, path: str) -> sitk.Transform:
    return _read_transform_recursive(h5file[path])


# ============================================================
# Main alignment (HDF5 output) with logging
# ============================================================


def align_cohort_to_global_grid(
    paths,
    out_paths,
    vol_key="data_masked",
    reference_index=0,
    spacing_mode="reference",  # "reference" or "min"
    do_affine_refine=True,
    pad_mm=0.0,
    registration_max_voxels=None,
    resample_slab_depth=8,
):
    assert len(paths) == len(out_paths)
    n = len(paths)
    t_all = time.time()

    log(
        f"Starting HDF5 alignment of {n} file(s). reference_index={reference_index}, spacing_mode={spacing_mode}, affine={do_affine_refine}, registration_max_voxels={registration_max_voxels}, resample_slab_depth={resample_slab_depth}"
    )

    # --- Load and preprocess reference ---
    log_step("Loading reference", reference_index, n, paths[reference_index])
    ref_vol, ref_spacing = load_h5_volume(
        paths[reference_index], vol_key
    )  # spacing_xyz
    log(
        f"  Reference spacing_xyz={_fmt_tuple(ref_spacing)}; raw shape={tuple(ref_vol.shape)}"
    )

    ref_vol_c, _, ref_off = crop_zero_planes(ref_vol)  # offset_zyx
    ref_origin = crop_offset_to_origin_xyz_mm(ref_off, ref_spacing)  # origin_xyz mm
    log(
        f"  Reference crop offset_zyx={ref_off}; cropped shape={tuple(ref_vol_c.shape)}; origin_xyz_mm={_fmt_tuple(ref_origin)}"
    )

    ref_mask_np, ref_fill_np = make_mask_and_filled(ref_vol_c)
    ref_img = to_sitk(ref_fill_np, ref_spacing, origin_xyz=ref_origin)
    ref_msk = to_sitk(ref_mask_np, ref_spacing, origin_xyz=ref_origin, is_mask=True)
    del ref_vol, ref_vol_c, ref_mask_np, ref_fill_np

    # Decide global spacing
    if spacing_mode == "reference":
        global_spacing = ref_spacing
        log(f"Global spacing set to reference spacing: {_fmt_tuple(global_spacing)}")
    elif spacing_mode == "min":
        log("Computing global spacing as min spacing across cohort...")
        spacings = []
        for i, p in enumerate(paths):
            log_step("  Reading spacing", i, n, p)
            _, sp = load_h5_volume(p, vol_key)
            spacings.append(np.array(sp, dtype=np.float64))
        global_spacing = tuple(np.min(np.vstack(spacings), axis=0).tolist())
        log(
            f"Global spacing set to min spacing across cohort: {_fmt_tuple(global_spacing)}"
        )
    else:
        raise ValueError("spacing_mode must be 'reference' or 'min'")

    transforms = [None] * n

    # bounds in reference space
    global_min = None
    global_max = None

    # include reference mask bounds
    b = mask_physical_bounds(ref_msk)
    if b is not None:
        bmin, bmax = b
        global_min = bmin if global_min is None else np.minimum(global_min, bmin)
        global_max = bmax if global_max is None else np.maximum(global_max, bmax)
        log(
            f"Reference mask bounds xyz mm: min={_fmt_tuple(global_min, 3)} max={_fmt_tuple(global_max, 3)}"
        )
    else:
        log("WARNING: Reference mask is empty after cropping.")

    # Keep legacy generation behaviour unless a caller explicitly requests a
    # voxel limit: transform estimation then uses the original-resolution
    # images and the original [4, 2, 1] SimpleITK pyramid.
    ref_reg_img, ref_shrink = shrink_for_registration(
        ref_img, registration_max_voxels
    )
    ref_reg_msk, ref_mask_shrink = shrink_for_registration(
        ref_msk, registration_max_voxels
    )
    assert ref_shrink == ref_mask_shrink
    log(
        f"Registration reference size_xyz={ref_img.GetSize()} -> {ref_reg_img.GetSize()} (shrink={ref_shrink})"
    )
    del ref_img, ref_msk

    # --- Estimate transforms to reference ---
    log("Estimating transforms to reference...")
    for i, p in enumerate(paths):
        if i == reference_index:
            transforms[i] = sitk.Transform(3, sitk.sitkIdentity)
            log_step("Transform", i, n, p, extra="identity (reference)")
            continue

        t0 = time.time()
        log_step("Preparing moving volume", i, n, p)
        vol, spacing = load_h5_volume(p, vol_key)  # spacing_xyz
        log(f"  Moving spacing_xyz={_fmt_tuple(spacing)}; raw shape={tuple(vol.shape)}")

        vol_c, _, off = crop_zero_planes(vol)  # offset_zyx
        origin = crop_offset_to_origin_xyz_mm(off, spacing)  # origin_xyz mm
        log(
            f"  Moving crop offset_zyx={off}; cropped shape={tuple(vol_c.shape)}; origin_xyz_mm={_fmt_tuple(origin)}"
        )

        msk_np, fill_np = make_mask_and_filled(vol_c)
        mov_img = to_sitk(fill_np, spacing, origin_xyz=origin)
        mov_msk = to_sitk(msk_np, spacing, origin_xyz=origin, is_mask=True)
        del vol, vol_c, msk_np, fill_np

        # Bounds must be measured on the original grid.  A reduced view is
        # only used when an explicit registration voxel limit is requested.
        mb = mask_physical_bounds(mov_msk)
        mov_reg_img, mov_shrink = shrink_for_registration(
            mov_img, registration_max_voxels
        )
        mov_reg_msk, mov_mask_shrink = shrink_for_registration(
            mov_msk, registration_max_voxels
        )
        assert mov_shrink == mov_mask_shrink
        log(
            f"  Registration moving size_xyz={mov_img.GetSize()} -> {mov_reg_img.GetSize()} (shrink={mov_shrink})"
        )
        del mov_img, mov_msk

        log("  Running rigid registration...")
        T_rigid = register_rigid(ref_reg_img, mov_reg_img, ref_reg_msk, mov_reg_msk)

        if do_affine_refine:
            log("  Running affine refinement...")
            T = register_affine_refine(
                ref_reg_img, mov_reg_img, ref_reg_msk, mov_reg_msk, T_rigid
            )
        else:
            T = T_rigid

        transforms[i] = T
        del mov_reg_img, mov_reg_msk

        # update global bounds using transformed moving mask bounds
        if mb is not None:
            mbmin, mbmax = mb
            tmin, tmax = transform_bounds(mbmin, mbmax, T)
            global_min = tmin if global_min is None else np.minimum(global_min, tmin)
            global_max = tmax if global_max is None else np.maximum(global_max, tmax)
            log(
                f"  Updated global bounds: min={_fmt_tuple(global_min, 3)} max={_fmt_tuple(global_max, 3)}"
            )
        else:
            log("  WARNING: Moving mask is empty after cropping; bounds not updated.")

        log(f"Transform estimation done in {time.time() - t0:.1f}s")

    if global_min is None or global_max is None:
        raise RuntimeError("All masks were empty; cannot build global grid.")

    # Optional padding
    global_min = global_min - float(pad_mm)
    global_max = global_max + float(pad_mm)
    if pad_mm:
        log(
            f"Applied pad_mm={pad_mm}. New bounds: min={_fmt_tuple(global_min, 3)} max={_fmt_tuple(global_max, 3)}"
        )

    # Build global grid
    extent_mm = global_max - global_min
    size_xyz = (
        np.ceil(extent_mm / np.array(global_spacing, dtype=np.float64)).astype(int) + 1
    )
    log(
        f"Building global grid: origin_xyz_mm={_fmt_tuple(global_min, 3)}, spacing_xyz={_fmt_tuple(global_spacing, 4)}, size_xyz={tuple(map(int, size_xyz))}"
    )
    if resample_slab_depth < 1:
        raise ValueError("resample_slab_depth must be at least 1")

    # --- Resample all into the global grid and write outputs ---
    log("Resampling all volumes into the global grid in Z-slabs and writing HDF5 outputs...")
    for i, (inp, outp) in enumerate(zip(paths, out_paths)):
        t0 = time.time()
        log_step("Resampling + writing", i, n, inp, extra=f"-> {Path(outp).name}")

        vol, spacing = load_h5_volume(inp, vol_key)
        vol_c, _, off = crop_zero_planes(vol)
        origin = crop_offset_to_origin_xyz_mm(off, spacing)

        msk_np, fill_np = make_mask_and_filled(vol_c)
        mov_img = to_sitk(fill_np, spacing, origin_xyz=origin)
        mov_msk = to_sitk(msk_np, spacing, origin_xyz=origin, is_mask=True)
        del vol, vol_c, msk_np, fill_np

        T = transforms[i]

        dx, dy, dz = global_spacing
        log("  Writing HDF5...")
        with h5py.File(outp, "w") as f:
            shape_zyx = tuple(map(int, size_xyz[::-1]))
            d = f.create_dataset(
                vol_key, shape=shape_zyx, dtype=np.float32, compression="gzip"
            )
            d.attrs["voxel_size"] = (float(dz), float(dy), float(dx))
            d.attrs["axis_order"] = "Z,Y,X"
            d.attrs["unit"] = "mm"

            # Global grid metadata
            d.attrs["global_origin_xyz_mm"] = tuple(map(float, global_min))
            d.attrs["global_spacing_xyz_mm"] = tuple(map(float, global_spacing))
            d.attrs["global_size_xyz"] = tuple(map(int, size_xyz))

            mask_dset = f.create_dataset(
                "mask", shape=shape_zyx, dtype=np.uint8, compression="gzip"
            )

            for z_start in range(0, shape_zyx[0], resample_slab_depth):
                z_stop = min(z_start + resample_slab_depth, shape_zyx[0])
                log(f"  Resampling Z slab [{z_start}:{z_stop})...")
                global_ref_slab = make_global_reference_slab(
                    global_min, global_spacing, size_xyz, z_start, z_stop
                )
                vol_global = resample(
                    mov_img, global_ref_slab, T, is_mask=False, default_value=0.0
                )
                msk_global = resample(
                    mov_msk, global_ref_slab, T, is_mask=True, default_value=0
                )
                vol_out = sitk.GetArrayFromImage(vol_global)
                msk_out = sitk.GetArrayFromImage(msk_global)
                vol_out[msk_out == 0] = np.nan
                d[z_start:z_stop] = vol_out
                mask_dset[z_start:z_stop] = msk_out
                del global_ref_slab, vol_global, msk_global, vol_out, msk_out

            log("  Saving transform to /transforms/to_reference")
            write_transform_h5(f, "/transforms/to_reference", transforms[i])

        log(f"Done in {time.time() - t0:.1f}s")

    log(f"All done. Total time: {time.time() - t_all:.1f}s")

    return {
        "reference_index": int(reference_index),
        "global_origin_xyz_mm": tuple(map(float, global_min)),
        "global_spacing_xyz_mm": tuple(map(float, global_spacing)),
        "global_size_xyz": tuple(map(int, size_xyz)),
        "transforms": transforms,
    }


# ============================================================
# transform other datasets to global grid
# ============================================================

import h5py
import SimpleITK as sitk


def load_h5_array(path, key):
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key} not found in {path}")
        return f[key][...]


def build_global_ref_from_aligned_h5(aligned_path, vol_key="data_masked"):
    """
    Rebuild the global reference image from the aligned output file's attrs.
    """
    with h5py.File(aligned_path, "r") as f:
        d = f[vol_key]
        origin = tuple(map(float, d.attrs["global_origin_xyz_mm"]))  # (x,y,z)
        spacing = tuple(map(float, d.attrs["global_spacing_xyz_mm"]))  # (x,y,z)
        size = tuple(map(int, d.attrs["global_size_xyz"]))  # (sx,sy,sz)
    return make_global_reference_image(origin, spacing, size), origin, spacing, size


def transform_dataset_to_global_grid(
    original_inp_h5: str,
    aligned_out_h5: str,
    dataset_key: str = "threshold_mask",
    vol_key_for_spacing: str = "data_masked",
    ref_key: str = "data",
    transform_path: str = "/transforms/to_reference",
    out_key: str | None = None,
):
    """
    Reads `dataset_key` from original_inp_h5, applies stored transform (from aligned_out_h5),
    and writes it back into aligned_out_h5 under out_key.

    Assumes:
      - dataset is Z,Y,X
      - implicit mask: zeros are background (OK for label/mask)
      - the transform was computed on the cropped volume (same crop must be applied here)
    """
    if out_key is None:
        out_key = f"aligned/{dataset_key}"

    # 1) Read dataset to be transformed
    arr = load_h5_array(original_inp_h5, dataset_key)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{dataset_key} must be 3D (Z,Y,X), got shape {arr.shape}")

    # 2) Use the SAME spacing conventions as the original registration
    #    (voxel_size read from ref_key)
    _, spacing_xyz = load_h5_volume(
        original_inp_h5, vol_key=vol_key_for_spacing, ref_key=ref_key
    )

    # 3) Crop zero-only planes (same logic), compute origin from crop offset
    arr_c, _, off = crop_zero_planes(arr)
    origin_xyz = crop_offset_to_origin_xyz_mm(off, spacing_xyz)

    # 4) Build SITK image (mask/labels => nearest-neighbor)
    #    If it's binary/label mask: treat as uint8
    arr_c_u8 = (
        (arr_c != 0).astype(np.uint8)
        if arr.dtype != np.uint8
        else arr_c.astype(np.uint8)
    )
    mov_msk = to_sitk(arr_c_u8, spacing_xyz, origin_xyz=origin_xyz, is_mask=True)

    # 5) Read stored transform from aligned output file
    with h5py.File(aligned_out_h5, "r") as f:
        T = read_transform_h5(f, transform_path)

    # 6) Rebuild the global reference image from aligned output attrs
    global_ref, origin_g, spacing_g, size_g = build_global_ref_from_aligned_h5(
        aligned_out_h5, vol_key=vol_key_for_spacing
    )

    # 7) Resample into global grid
    msk_global = resample(mov_msk, global_ref, T, is_mask=True, default_value=0)
    out = sitk.GetArrayFromImage(msk_global).astype(np.uint8)  # Z,Y,X

    # 8) Write into aligned output file
    with h5py.File(aligned_out_h5, "a") as f:
        if out_key in f:
            del f[out_key]
        d = f.create_dataset(out_key, data=out, compression="gzip")
        d.attrs["axis_order"] = "Z,Y,X"
        d.attrs["unit"] = "mm"
        # store the same global grid attrs for convenience
        d.attrs["global_origin_xyz_mm"] = origin_g
        d.attrs["global_spacing_xyz_mm"] = spacing_g
        d.attrs["global_size_xyz"] = size_g

    return out


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Register HDF5 volumes into a shared global grid."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files.")
    parser.add_argument(
        "--outputs", nargs="+", required=True, help="Output _global.hdf5 files."
    )
    parser.add_argument("--vol-key", default="data_masked")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument(
        "--spacing-mode", choices=["reference", "min"], default="reference"
    )
    parser.add_argument("--no-affine-refine", action="store_true")
    parser.add_argument("--pad-mm", type=float, default=0.0)
    parser.add_argument(
        "--registration-max-voxels",
        type=int,
        default=None,
        help="Optional maximum voxel count per image used during transform estimation.",
    )
    parser.add_argument(
        "--resample-slab-depth",
        type=int,
        default=8,
        help="Number of Z slices resampled and written at a time.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if len(args.inputs) != len(args.outputs):
        raise SystemExit("--inputs and --outputs must contain the same number of paths")
    align_cohort_to_global_grid(
        paths=args.inputs,
        out_paths=args.outputs,
        vol_key=args.vol_key,
        reference_index=args.reference_index,
        spacing_mode=args.spacing_mode,
        do_affine_refine=not args.no_affine_refine,
        pad_mm=args.pad_mm,
        registration_max_voxels=args.registration_max_voxels,
        resample_slab_depth=args.resample_slab_depth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
