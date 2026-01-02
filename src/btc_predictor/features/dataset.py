from typing import Dict, List

import numpy as np
import pandas as pd

from btc_predictor.config import get_base_frequency, get_horizon_labels, get_horizon_map, horizon_to_steps


def build_triple_barrier_targets(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = raw_df[["timestamp", "close", "high", "low"]].copy()
    base_frequency = get_base_frequency(cfg)
    
    tb_cfg = cfg["targets"].get("triple_barrier", {})
    profit_pct = tb_cfg.get("profit", 0.02)
    stop_pct = tb_cfg.get("stop", 0.01)
    horizon_str = tb_cfg.get("barrier_horizon", "12h")
    
    steps = horizon_to_steps(horizon_str, base_frequency)
    
    labels = np.zeros(len(df), dtype=int)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    
    # Fast iteration
    for i in range(n - steps):
        current_close = closes[i]
        window_highs = highs[i+1 : i+steps+1]
        window_lows = lows[i+1 : i+steps+1]
        
        take_profit = current_close * (1 + profit_pct)
        stop_loss = current_close * (1 - stop_pct)
        
        hit_tp_mask = window_highs >= take_profit
        hit_sl_mask = window_lows <= stop_loss
        
        has_tp = hit_tp_mask.any()
        has_sl = hit_sl_mask.any()
        
        if has_tp and not has_sl:
            labels[i] = 1
        elif has_sl and not has_tp:
            labels[i] = -1
        elif has_tp and has_sl:
            first_tp_idx = np.argmax(hit_tp_mask)
            first_sl_idx = np.argmax(hit_sl_mask)
            if first_tp_idx <= first_sl_idx:
                labels[i] = 1
            else:
                labels[i] = -1
        else:
            labels[i] = 0
            
    df[f"y_{horizon_str}"] = labels
    df[f"target_time_{horizon_str}"] = df["timestamp"] + pd.Timedelta(horizon_str)
    return df[["timestamp", f"y_{horizon_str}", f"target_time_{horizon_str}"]]


def build_targets(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if cfg["targets"].get("type") == "classification":
        return build_triple_barrier_targets(raw_df, cfg)

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
