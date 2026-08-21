import pickle

import h5py
import numpy as np
import pytest

from basalt_processing.amira_hdf5 import am_to_hdf5, build_parser
from basalt_processing.hdf5_amira import write_amira_lattice


def test_amira_hdf5_parser_accepts_required_paths(tmp_path):
    input_am = tmp_path / "input.am"
    output_h5 = tmp_path / "output.hdf5"
    args = build_parser().parse_args([str(input_am), str(output_h5), "--dset", "data"])
    assert args.input_am == input_am
    assert args.output_h5 == output_h5
    assert args.dset == "data"


def test_amira_hdf5_parser_rejects_missing_paths():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_am_to_hdf5_writes_pickled_header_on_created_dataset(tmp_path):
    input_am = tmp_path / "input.am"
    output_h5 = tmp_path / "output.hdf5"
    write_amira_lattice(input_am, np.arange(24, dtype=np.uint16).reshape(2, 3, 4))

    am_to_hdf5(input_am, output_h5, dset_name="volume")

    with h5py.File(output_h5, "r") as h5:
        assert "amira_header_pickled" not in h5.attrs
        raw_header = h5["volume"].attrs["amira_header_pickled"]
        parsed_header = pickle.loads(bytes(raw_header))
        assert parsed_header
