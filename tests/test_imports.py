"""Phase-0 smoke tests: the new package tree imports and config loads."""

import importlib

import pytest

PACKAGES = [
    "src",
    "src.data",
    "src.forecasters",
    "src.regimes",
    "src.conformal",
    "src.eval",
    "src.experiments",
    "src.utils",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_imports(pkg):
    importlib.import_module(pkg)


def test_base_config_loads():
    from src.utils.config import load_config

    cfg = load_config()
    assert cfg["project_name"] == "rccv"
    assert cfg["data"]["universe"]["size"] == 100
    assert cfg["evaluation"]["alpha_two_sided"] == 0.10


def test_seeding_is_deterministic():
    import numpy as np

    from src.utils.seeding import seed_everything

    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert (a == b).all()
