"""One convention for every pluggable piece: models and datasets alike.

A module under ``src/models/`` or ``src/datasets/`` declares two names:

    def default_HPs(cfg):   # may read the whole config for data-dependent sizes
        return OmegaConf.create({...})

    MODEL = MyNet           # or DATASET = MyDataset

Defaults live with the implementation, ``<section>.params`` in the experiment config
carries only what you want changed, and the two are merged here. An unknown key in
``params`` is an error rather than a silent no-op. Both builders write the merged
params back into the config, so what gets logged is what was actually built.
"""

import importlib

from omegaconf import OmegaConf


PACKAGES = {"model": "src.models", "data": "src.datasets"}  # config section -> where to look


def resolve_params(section_name, cfg):
    """Fill ``cfg.<section_name>.params`` with the module's defaults under what is set.
    """
    section = cfg[section_name]
    module = importlib.import_module(f"{PACKAGES[section_name]}.{section.name}")
    defaults = module.default_HPs(cfg)
    overrides = OmegaConf.create(section.get("params") or {})

    unknown = set(overrides.keys()) - set(defaults.keys())
    if unknown:
        raise ValueError(
            f"params has no such key(s) for {section.name!r}: {sorted(unknown)}; "
            f"it takes {sorted(defaults.keys())}"
        )
    section.params = OmegaConf.merge(defaults, overrides)
    return module


def build_model(cfg):
    return resolve_params("model", cfg).MODEL(cfg.model.params)


def build_dataset(cfg):
    return resolve_params("data", cfg).DATASET(cfg.data.params)
