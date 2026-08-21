from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import pickle

import h5py
import numpy as np

from basalt_processing.amira_to_np.amira_helper import np_to_amira
from basalt_processing.paths import ensure_parent


SUPPORTED_DTYPES = {
    np.dtype("uint8"): "byte",
    np.dtype("int16"): "short",
    np.dtype("uint16"): "ushort",
    np.dtype("int32"): "int",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
}


def choose_dataset(h5: h5py.File, requested: str | None = None) -> str:
    """Choose a 3D dataset from an HDF5 file."""
    if requested:
        if requested not in h5:
            raise KeyError(f"Dataset {requested!r} not found")
        return requested
    for candidate in ("data_masked", "data"):
        if candidate in h5:
            return candidate
    datasets = [name for name, value in h5.items() if isinstance(value, h5py.Dataset)]
    if not datasets:
        raise KeyError("No top-level datasets found")
    return datasets[0]


def amira_dtype_name(dtype: np.dtype) -> str:
    """Return the AmiraMesh scalar name for a NumPy dtype."""
    normalized = np.dtype(dtype)
    if normalized not in SUPPORTED_DTYPES:
        raise TypeError(f"Unsupported Amira dtype: {normalized}")
    return SUPPORTED_DTYPES[normalized]


def _voxel_size_zyx_from_attrs(dset: h5py.Dataset) -> tuple[float, float, float] | None:
    if "voxel_size" in dset.attrs:
        values = tuple(float(v) for v in dset.attrs["voxel_size"])
        if len(values) == 3:
            return values
    if "voxel_size_xyz" in dset.attrs:
        x, y, z = (float(v) for v in dset.attrs["voxel_size_xyz"])
        return z, y, x
    return None


def _format_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return "0" if text == "-0" else text


def build_amira_header(data: np.ndarray, voxel_size_zyx: Sequence[float] | None = None) -> bytes:
    """Build a simple binary AmiraMesh lattice header for a Z,Y,X array."""
    if data.ndim != 3:
        raise ValueError(f"Amira export requires a 3D array, got shape {data.shape}")
    z_size, y_size, x_size = data.shape
    z_res, y_res, x_res = (1.0, 1.0, 1.0) if voxel_size_zyx is None else tuple(float(v) for v in voxel_size_zyx)
    x_max = (x_size - 1) * x_res
    y_max = (y_size - 1) * y_res
    z_max = (z_size - 1) * z_res
    dtype_name = amira_dtype_name(data.dtype)
    header = "\n".join([
        "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1",
        "",
        f"define Lattice {x_size} {y_size} {z_size}",
        "",
        "Parameters {",
        f"    BoundingBox 0 {_format_float(x_max)} 0 {_format_float(y_max)} 0 {_format_float(z_max)},",
        "    CoordType \"uniform\"",
        "}",
        "",
        f"Lattice {{ {dtype_name} Data }} @1",
        "",
        "@1",
        "",
    ])
    return header.encode("ascii")


def write_amira_lattice(
    output_path: str | Path,
    data: np.ndarray,
    voxel_size_zyx: Sequence[float] | None = None,
) -> Path:
    """Write a 3D NumPy array as a simple binary AmiraMesh lattice."""
    out_path = ensure_parent(output_path)
    array = np.asarray(data)
    header = build_amira_header(array, voxel_size_zyx)
    with out_path.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(array).tobytes(order="C"))
        handle.write(b"\n")
    return out_path


def hdf5_to_amira(
    input_h5: str | Path,
    output_am: str | Path,
    dataset: str | None = None,
    dtype: str | None = None,
    header_source_dataset: str = "data",
) -> Path:
    """Convert one HDF5 dataset using its stored original Amira header."""
    input_path = Path(input_h5)
    with h5py.File(input_path, "r") as h5:
        dataset_name = choose_dataset(h5, dataset)
        dset = h5[dataset_name]
        data = dset[...]
        if dtype is not None:
            data = data.astype(np.dtype(dtype), copy=False)
        if header_source_dataset not in h5:
            raise KeyError(
                f"Header source dataset {header_source_dataset!r} not found"
            )
        header_source = h5[header_source_dataset]
        if "amira_header_pickled" not in header_source.attrs:
            raise KeyError(
                "Attribute 'amira_header_pickled' not found on dataset "
                f"{header_source_dataset!r}"
            )
        headers = pickle.loads(bytes(header_source.attrs["amira_header_pickled"]))

    output_path = ensure_parent(output_am)
    np_to_amira(output_path, [data], headers[0])
    return output_path


def discover_hdf5_inputs(input_path: str | Path, recursive: bool = False) -> list[Path]:
    """Return HDF5 files from one file or directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    files = sorted(path.glob(pattern))
    files.extend(sorted(path.glob("**/*.h5" if recursive else "*.h5")))
    return files


def convert_many(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    dataset: str | None = None,
    suffix: str = ".am",
    dtype: str | None = None,
    header_source_dataset: str = "data",
) -> list[Path]:
    """Convert multiple HDF5 files into one output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for input_file in inputs:
        input_path = Path(input_file)
        output_path = out_dir / f"{input_path.stem}{suffix}"
        outputs.append(
            hdf5_to_amira(
                input_path,
                output_path,
                dataset=dataset,
                dtype=dtype,
                header_source_dataset=header_source_dataset,
            )
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HDF5 volumes back to Amira .am files.")
    parser.add_argument("input", type=Path, help="Input HDF5 file or directory.")
    parser.add_argument("output", type=Path, nargs="?", help="Output .am file for single-file mode.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for directory or batch mode.")
    parser.add_argument("--dataset", help="Dataset name to export. Defaults to data_masked, then data.")
    parser.add_argument(
        "--header-source-dataset",
        default="data",
        help="Dataset containing the pickled original Amira header.",
    )
    parser.add_argument("--suffix", default=".am", help="Output suffix for batch mode.")
    parser.add_argument("--dtype", help="Optional NumPy dtype cast before export, for example uint16.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = discover_hdf5_inputs(args.input, recursive=args.recursive)
    if not inputs:
        raise SystemExit(f"No HDF5 inputs found under {args.input}")
    if len(inputs) == 1 and args.output is not None:
        hdf5_to_amira(
            inputs[0],
            args.output,
            dataset=args.dataset,
            dtype=args.dtype,
            header_source_dataset=args.header_source_dataset,
        )
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for directory or batch conversion")
    convert_many(
        inputs,
        args.output_dir,
        dataset=args.dataset,
        suffix=args.suffix,
        dtype=args.dtype,
        header_source_dataset=args.header_source_dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
