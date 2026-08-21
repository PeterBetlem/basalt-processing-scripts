from pathlib import Path
import pickle

import h5py
import numpy as np
import pytest

from basalt_processing.amira_hdf5 import am_to_hdf5
from basalt_processing.hdf5_amira import (
    amira_dtype_name,
    build_amira_header,
    build_parser,
    choose_dataset,
    hdf5_to_amira,
)


def test_choose_dataset_prefers_data_masked(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((1, 2, 3), dtype=np.uint8))
        h5.create_dataset("data_masked", data=np.ones((1, 2, 3), dtype=np.uint8))
    with h5py.File(h5_path, "r") as h5:
        assert choose_dataset(h5) == "data_masked"


def test_choose_dataset_uses_requested_dataset(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("mask", data=np.ones((1, 2, 3), dtype=np.uint8))
    with h5py.File(h5_path, "r") as h5:
        assert choose_dataset(h5, "mask") == "mask"


def test_choose_dataset_raises_for_missing_dataset(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("mask", data=np.ones((1, 2, 3), dtype=np.uint8))
    with h5py.File(h5_path, "r") as h5:
        with pytest.raises(KeyError, match="not found"):
            choose_dataset(h5, "data")


def test_amira_dtype_name_maps_supported_types():
    assert amira_dtype_name(np.dtype("uint8")) == "byte"
    assert amira_dtype_name(np.dtype("int16")) == "short"
    assert amira_dtype_name(np.dtype("uint16")) == "ushort"
    assert amira_dtype_name(np.dtype("float32")) == "float"


def test_hdf5_to_amira_preserves_original_header_for_masked_data(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    out_path = tmp_path / "volume.am"
    roundtrip_h5 = tmp_path / "roundtrip.hdf5"
    data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    header = build_amira_header(data, (0.3, 0.2, 0.1)).replace(
        b'    CoordType "uniform"',
        b'    Content "preserved-original-header",\n    CoordType "uniform"',
    )
    header = header.rsplit(b"\n@1\n", maxsplit=1)[0]
    with h5py.File(h5_path, "w") as h5:
        reference = h5.create_dataset("data", data=data)
        reference.attrs["amira_header_pickled"] = np.void(
            pickle.dumps([np.frombuffer(header, dtype=np.int8)])
        )
        h5.create_dataset("data_masked", data=data)

    result = hdf5_to_amira(h5_path, out_path)

    assert result == out_path
    raw = out_path.read_bytes()
    assert raw.startswith(b"# AmiraMesh BINARY-LITTLE-ENDIAN 2.1")
    assert b'Content "preserved-original-header"' in raw
    assert raw.endswith(data.tobytes(order="C") + b"\n")

    am_to_hdf5(out_path, roundtrip_h5)
    with h5py.File(roundtrip_h5, "r") as h5:
        restored_headers = pickle.loads(
            bytes(h5["data"].attrs["amira_header_pickled"])
        )
    assert b'Content "preserved-original-header"' in restored_headers[0].tobytes()


def test_hdf5_to_amira_requires_pickled_header_on_header_source_dataset(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    out_path = tmp_path / "volume.am"
    data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("data", data=data)
        h5.create_dataset("data_masked", data=data)

    with pytest.raises(KeyError, match="amira_header_pickled"):
        hdf5_to_amira(h5_path, out_path)


def test_parser_accepts_directory_mode(tmp_path):
    args = build_parser().parse_args([
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "am"),
        "--dataset",
        "data",
        "--recursive",
    ])
    assert args.input == tmp_path
    assert args.output_dir == tmp_path / "am"
    assert args.dataset == "data"
    assert args.recursive is True
