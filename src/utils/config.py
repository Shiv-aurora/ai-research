"""Config loading for the RCCV project.

Base config lives in conf/base/config.yaml; experiment YAMLs under
conf/experiments/ are shallow overrides merged on top of it.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_CONFIG = PROJECT_ROOT / "conf" / "base" / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(experiment_yaml: str | Path | None = None) -> dict:
    """Load base config, optionally merged with an experiment override."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if experiment_yaml is not None:
        with open(experiment_yaml) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg
