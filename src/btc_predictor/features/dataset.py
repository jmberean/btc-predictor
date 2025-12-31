from typing import Dict, List

import numpy as np
import pandas as pd

from btc_predictor.config import get_base_frequency, get_horizon_labels, get_horizon_map, horizon_to_steps


def build_targets(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = raw_df[["timestamp", "close"]].copy()
    base_frequency = get_base_frequency(cfg)
    horizon_map = get_horizon_map(cfg)
    
    # Optional Volatility Normalization
    target_mode = cfg["targets"].get("mode", "raw") # "raw" or "zscore"
    vol_window = cfg["targets"].get("vol_window", 168) # 1 week default

    if target_mode == "zscore":
        # Calculate rolling volatility for normalization
        log_ret = np.log(df["close"] / df["close"].shift(1))
        rolling_vol = log_ret.rolling(vol_window).std()
    else:
        rolling_vol = None

    for label, h in horizon_map.items():
        steps = horizon_to_steps(h, base_frequency)
        raw_return = np.log(df["close"].shift(-steps) / df["close"])
        
        if target_mode == "zscore" and rolling_vol is not None:
            # Normalize by sqrt(steps) because volatility scales with square root of time
            df[f"y_{label}"] = raw_return / (rolling_vol * np.sqrt(steps))
        else:
            df[f"y_{label}"] = raw_return
            
        df[f"target_time_{label}"] = df["timestamp"] + h
    
    return df.drop(columns=["close"])


def build_supervised_dataset(raw_df: pd.DataFrame, feature_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    targets = build_targets(raw_df, cfg)
    df = feature_df.merge(targets, on="timestamp", how="left")
    horizon_map = get_horizon_map(cfg)
    df["max_target_time"] = df["timestamp"] + max(horizon_map.values())
    df = df.dropna(subset=[f"y_{label}" for label in horizon_map.keys()])
    df = df.reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame, horizon_labels: List[str], feature_cols: List[str]):
    x = df[feature_cols]
    y = {label: df[f"y_{label}"].values for label in horizon_labels}
    return x, y
