from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import h5py
import numpy as np

from basalt_processing.amira_to_np.amira_helper import amira_to_np
from basalt_processing.amira_to_np.mesh import get_voxel_spacing
from basalt_processing.paths import ensure_parent


def am_to_hdf5(
    input_am: str | Path,
    output_h5: str | Path,
    dset_name: str = "data",
    gzip_level: int = 4,
    extension: str = ".am",
) -> None:
    input_path = Path(input_am)
    output_path = ensure_parent(output_h5)

    data_parts, header0 = amira_to_np(input_path.as_posix())
    headers = [header0]
    if len(data_parts) > 1:
        headers.extend(data_parts[1:])
    data = data_parts[0]

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array from Amira loader, got {type(data)!r}")

    try:
        x_res, y_res, z_res = get_voxel_spacing(headers, extension)
        voxel_size = np.array([z_res, y_res, x_res], dtype=np.float32)
    except Exception:
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        x_res, y_res, z_res = voxel_size[::-1]

    z_size, y_size, x_size = data.shape
    chunks = (min(64, z_size), min(64, y_size), min(64, x_size))

    with h5py.File(output_path, "w") as h5:
        dset = h5.create_dataset(
            dset_name,
            data=data,
            dtype=data.dtype,
            chunks=chunks,
            compression="gzip",
            compression_opts=int(gzip_level),
        )
        dset.attrs["voxel_size"] = voxel_size
        dset.attrs["voxel_size_xyz"] = np.array([x_res, y_res, z_res], dtype=np.float32)
        dset.attrs["axis_order"] = "Z,Y,X"
        dset.attrs["unit"] = "mm"
        dset.attrs["source_file"] = input_path.as_posix()
        dset.attrs["amira_extension"] = extension
        dset.attrs["amira_header_pickled"] = np.void(pickle.dumps(headers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an Amira/Avizo .am file to HDF5.")
    parser.add_argument("input_am", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("--dset", default="data", help="Dataset name inside the HDF5 file.")
    parser.add_argument("--gzip", type=int, default=4, help="Gzip compression level, 0-9.")
    parser.add_argument("--extension", default=".am", help="Amira extension used for voxel metadata parsing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    am_to_hdf5(args.input_am, args.output_h5, args.dset, args.gzip, args.extension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
