from pathlib import Path

import pytest

from vibnet.schema import load_config


@pytest.fixture
def cfgs() -> list[Path]:
    cfg_path = Path(".").absolute() / "cfgs"
    files = [
        cfg_path / "dnn-example.yaml",
        cfg_path / "knn-example.yaml",
        cfg_path / "randomforest-example.yaml",
        cfg_path / "1nn-example.yaml",
    ]
    return files


def test_schemas(cfgs: list[Path]):
    for p in cfgs:
        load_config(p)


def test_deep_schema():
    cwru = Path(".").absolute() / "cfgs" / "dnn-example.yaml"
    config_type, _ = load_config(cwru)
    assert config_type == "deep", f"Config is type '{config_type}', not 'deep'"


def test_classic_schema():
    rf = Path(".").absolute() / "cfgs" / "randomforest-example.yaml"
    config_type, _ = load_config(rf)
    assert config_type == "classic", f"Config is type '{config_type}', not 'classic'"
