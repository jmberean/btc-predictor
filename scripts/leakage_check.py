import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from btc_predictor.config import get_horizon_labels, load_config
from btc_predictor.data.ingestion import fetch_ohlcv_ccxt, load_ohlcv_csv
from btc_predictor.evaluation.leakage import time_travel_check
from btc_predictor.evaluation.walk_forward import generate_walk_forward_splits
from btc_predictor.features.dataset import build_supervised_dataset
from btc_predictor.features.engineering import build_feature_frame, feature_columns


def _load_data(cfg):
    data_cfg = cfg["data"]
    if data_cfg["source"] == "ccxt":
        return fetch_ohlcv_ccxt(
            exchange_id=data_cfg["exchange"],
            symbol=data_cfg["symbol"],
            timeframe=data_cfg["timeframe"],
            start=data_cfg.get("start"),
            end=data_cfg.get("end"),
        )
    return load_ohlcv_csv(data_cfg["csv_path"], tz=data_cfg.get("timezone", "UTC"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = _load_data(cfg)
    feature_df = build_feature_frame(raw_df, cfg)
    dataset = build_supervised_dataset(raw_df, feature_df, cfg)
    dataset = dataset.sort_values("prediction_time").reset_index(drop=True)

    splits = generate_walk_forward_splits(dataset, cfg)
    if not splits:
        raise ValueError("No walk-forward splits available")

    train_idx, test_idx = splits[0]
    train_df = dataset.loc[train_idx]
    test_df = dataset.loc[test_idx]

    feature_cols = feature_columns(dataset)
    horizon_label = get_horizon_labels(cfg)[0]
    target_col = f"y_{horizon_label}"

    results = time_travel_check(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        params=cfg.get("lightgbm", {}),
    )
    print(results)


if __name__ == "__main__":
    main()