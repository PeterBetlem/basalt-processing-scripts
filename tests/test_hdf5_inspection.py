import pickle

import h5py
import numpy as np

from basalt_processing.hdf5_inspection import (
    describe_hdf5,
    find_pickled_amira_headers,
    is_pickled_amira_header,
    raw_hdf5_attribute,
    render_amira_header,
    unpickle_amira_header,
)


def make_hdf5_with_amira_header(path):
    header_text = "# Avizo BINARY-LITTLE-ENDIAN 3.0\ndefine Lattice 3 2 1\n"
    with h5py.File(path, "w") as h5:
        dataset = h5.create_dataset("scan/volume", data=np.array([1, 2, 3], dtype=np.uint16))
        dataset.attrs["scanner"] = "Xradia"
        dataset.attrs["amira_header_pickled"] = np.void(
            pickle.dumps([np.frombuffer(header_text.encode("utf-8"), dtype=np.int8)])
        )
    return header_text


def test_is_pickled_amira_header_matches_descriptive_attribute_name():
    assert is_pickled_amira_header("amira_header_pickled")
    assert not is_pickled_amira_header("scanner")


def test_describe_hdf5_omits_pickled_header_value(tmp_path, capsys):
    h5_path = tmp_path / "volume.hdf5"
    make_hdf5_with_amira_header(h5_path)

    describe_hdf5(h5_path)

    output = capsys.readouterr().out
    assert "Dataset: scan/volume" in output
    assert "attribute: scanner = Xradia" in output
    assert "attribute: amira_header_pickled (pickled Amira header omitted)" in output
    assert "Avizo BINARY-LITTLE-ENDIAN" not in output


def test_header_helpers_unpickle_and_render_amira_text(tmp_path):
    h5_path = tmp_path / "volume.hdf5"
    expected_header = make_hdf5_with_amira_header(h5_path)

    assert find_pickled_amira_headers(h5_path) == [("scan/volume", "amira_header_pickled")]
    raw_value = raw_hdf5_attribute(h5_path, "scan/volume", "amira_header_pickled")
    assert raw_value is not None

    header = unpickle_amira_header(h5_path, "scan/volume", "amira_header_pickled")
    assert render_amira_header(header) == expected_header
