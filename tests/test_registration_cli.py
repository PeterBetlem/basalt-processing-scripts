import pytest

from basalt_processing.registration import build_parser, main


def test_registration_parser_accepts_matching_inputs_outputs():
    args = build_parser().parse_args([
        "--inputs", "a.hdf5", "b.hdf5",
        "--outputs", "a._global.hdf5", "b._global.hdf5",
        "--vol-key", "data_masked",
    ])
    assert args.inputs == ["a.hdf5", "b.hdf5"]
    assert args.outputs == ["a._global.hdf5", "b._global.hdf5"]
    assert args.vol_key == "data_masked"


def test_registration_main_rejects_mismatched_input_output_counts():
    with pytest.raises(SystemExit, match="same number"):
        main(["--inputs", "a.hdf5", "b.hdf5", "--outputs", "a._global.hdf5"])
