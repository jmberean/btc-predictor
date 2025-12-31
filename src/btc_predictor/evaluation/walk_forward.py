from typing import Dict, List, Tuple

import pandas as pd

from btc_predictor.config import get_horizons


def generate_walk_forward_splits(df: pd.DataFrame, cfg: Dict) -> List[Tuple[List[int], List[int]]]:
    wf_cfg = cfg["training"]["walk_forward"]
    scheme = wf_cfg.get("scheme", "expanding")
    train_window = pd.Timedelta(days=wf_cfg["train_window_days"])
    test_window = pd.Timedelta(days=wf_cfg["test_window_days"])
    step = pd.Timedelta(days=wf_cfg["step_days"])
    
    # NEW: Rigorous Gaps
    embargo_hours = wf_cfg.get("embargo_hours", 0)
    purge_hours = wf_cfg.get("purge_hours", 0) # Gap before test set
    
    embargo = pd.Timedelta(hours=embargo_hours)
    purge = pd.Timedelta(hours=purge_hours)

    horizons = get_horizons(cfg)
    max_horizon = max(horizons)
    # Ensure embargo is at least the max forecast horizon
    embargo = max(embargo, max_horizon)

    df = df.sort_values("prediction_time")
    start_time = df["prediction_time"].min()
    end_time = df["prediction_time"].max()

    splits = []
    window_start = start_time + train_window

    while window_start + test_window <= end_time:
        # Purging: Move the train_end back to prevent overlap from lagging features
        train_end = window_start - purge
        
        test_start = window_start + embargo
        test_end = test_start + test_window

        if scheme == "rolling":
            train_start = window_start - train_window
        else:
            train_start = start_time

        train_mask = (df["prediction_time"] >= train_start) & (df["prediction_time"] < train_end)
        # Point-in-time check: Ensure no target in train can see into test
        train_mask &= df["max_target_time"] <= train_end
        
        test_mask = (df["prediction_time"] >= test_start) & (df["prediction_time"] < test_end)

        train_idx = df.index[train_mask].tolist()
        test_idx = df.index[test_mask].tolist()
        if train_idx and test_idx:
            splits.append((train_idx, test_idx))

        window_start += step

    return splits
