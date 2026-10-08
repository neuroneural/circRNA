"""YAML config with CLI overrides.

Deliberately not Hydra: `abide` does not ship it, and one file of plain YAML plus
dotted `key=value` overrides covers what we need.

    python -m src.train --config conf/experiments/hamd_transfer.yaml \
           site.mode=combat data.cohort=all train.n_repeats=1
"""

import argparse
import copy
import os

import yaml

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "conf", "base.yaml")


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if (k in out and isinstance(out[k], dict)
                                            and isinstance(v, dict)) else copy.deepcopy(v)
    return out


def _coerce(s):
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_coerce(x) for x in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


class Cfg(dict):
    """dict with attribute access, so cfg.data.cohort reads naturally."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load(argv=None, extra_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="experiment YAML, merged over base")
    ap.add_argument("--base", default=BASE)
    for a in (extra_args or []):
        ap.add_argument(a, default=None)
    args, rest = ap.parse_known_args(argv)

    with open(args.base) as f:
        cfg = yaml.safe_load(f)
    if args.config:
        with open(args.config) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    for tok in rest:
        if "=" not in tok:
            raise SystemExit(f"unrecognised argument: {tok}\n"
                             "overrides must look like  site.mode=combat")
        k, v = tok.split("=", 1)
        _set_dotted(cfg, k.lstrip("-"), _coerce(v))

    cfg["_config_file"] = args.config or "(base only)"
    return Cfg(cfg), args


def describe(cfg):
    d = cfg["data"]; s = cfg["site"]; l = cfg["label"]; m = cfg["model"]
    lines = [
        f"config      : {cfg.get('_config_file')}",
        f"cohort      : {d['cohort']}   groups {d['groups']}",
        f"modalities  : {d['modalities']}  ({d['representation']})",
        f"missing mods: {'allowed (masked)' if d['allow_missing_modalities'] else 'complete-case'}",
        f"site        : {s['mode']}",
        f"label       : {l['target']}" + (f"   hamd_mode {l['hamd_mode']}"
                                          if str(l['target']).startswith('hamd') else ""),
        f"fusion      : {m['fusion']['mode']}   smart_init {m['smart_init']}",
    ]
    return "\n".join("  " + x for x in lines)
