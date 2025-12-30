import os
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config is empty: {path}")
    return cfg


def parse_timedelta(value: str) -> pd.Timedelta:
    return pd.Timedelta(value)


def get_horizon_labels(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["targets"]["horizons"])


def get_horizon_map(cfg: Dict[str, Any]) -> Dict[str, pd.Timedelta]:
    return {label: parse_timedelta(label) for label in get_horizon_labels(cfg)}


def get_horizons(cfg: Dict[str, Any]) -> List[pd.Timedelta]:
    return list(get_horizon_map(cfg).values())


def get_base_frequency(cfg: Dict[str, Any]) -> pd.Timedelta:
    return parse_timedelta(cfg["features"]["base_frequency"])


def horizon_to_steps(horizon: pd.Timedelta, base_frequency: pd.Timedelta) -> int:
    steps = int(horizon / base_frequency)
    if steps <= 0:
        raise ValueError(f"Invalid horizon {horizon} for base frequency {base_frequency}")
    return steps
