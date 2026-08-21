from pathlib import Path

from basalt_processing.paths import ensure_parent, load_config, resolve_path


def test_load_config_returns_empty_dict_without_path():
    assert load_config(None) == {}


def test_resolve_path_uses_base_dir_for_relative_paths(tmp_path):
    result = resolve_path("data/input.txt", base_dir=tmp_path)
    assert result == (tmp_path / "data" / "input.txt").resolve()


def test_ensure_parent_creates_output_parent(tmp_path):
    out = ensure_parent(tmp_path / "nested" / "file.csv")
    assert out == tmp_path / "nested" / "file.csv"
    assert Path(out).parent.exists()
