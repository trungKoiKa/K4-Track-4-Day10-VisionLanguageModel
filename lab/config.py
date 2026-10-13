"""Đọc configs/eN.yaml và áp dụng --set a.b=value (dùng cho E4)."""
from pathlib import Path

import yaml

DEFAULTS = {"data_size": 1.0, "coord_scale": "norm1000"}


def load_config(path, overrides=()):
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text(encoding="utf-8"))}
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
