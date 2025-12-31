import json
import os
from typing import Dict, List, Optional

import joblib
import pandas as pd

from btc_predictor.config import get_horizon_labels, get_horizon_map, load_config
from btc_predictor.data.binance_bulk import load_binance_bulk_klines
from btc_predictor.data.external import load_external_features
from btc_predictor.data.ingestion import fetch_ohlcv_ccxt, load_ohlcv_csv
from btc_predictor.features.engineering import build_feature_frame, feature_columns


def _load_data(cfg: Dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    if data_cfg["source"] == "ccxt":
        return fetch_ohlcv_ccxt(
            exchange_id=data_cfg["exchange"],
            symbol=data_cfg["symbol"],
            timeframe=data_cfg["timeframe"],
            start=data_cfg.get("start"),
            end=data_cfg.get("end"),
        )
    if data_cfg["source"] == "binance_bulk":
        bulk_cfg = data_cfg["binance_bulk"]
        return load_binance_bulk_klines(
            bulk_cfg["klines_path"],
            timeframe=bulk_cfg.get("timeframe", data_cfg["timeframe"]),
            tz=data_cfg.get("timezone", "UTC"),
            start=data_cfg.get("start"),
            end=data_cfg.get("end"),
        )
    return load_ohlcv_csv(data_cfg["csv_path"], tz=data_cfg.get("timezone", "UTC"))


def run_inference(cfg_path: str, model_paths: List[str], asof: str, output_path: str, weights: Optional[List[float]] = None) -> str:
    cfg = load_config(cfg_path)
    
    if asof:
        cfg["data"]["end"] = asof
        if "external_data" in cfg:
            for k in cfg["external_data"]:
                if isinstance(cfg["external_data"][k], dict):
                    cfg["external_data"][k]["end"] = asof

    raw_df = _load_data(cfg)
    external_dfs = load_external_features(cfg)
    feature_df = build_feature_frame(raw_df, cfg, external_dfs=external_dfs)
    feature_cols = feature_columns(feature_df)

    if asof:
        asof_ts = pd.Timestamp(asof, tz="UTC")
    else:
        asof_ts = pd.Timestamp.now(tz="UTC")

    available = feature_df[feature_df["prediction_time"] <= asof_ts].copy()
    if available.empty:
        raise ValueError("No features available at the requested asof time")

    x = available[feature_cols].values
    x_last = x[-1:]

    horizon_labels = get_horizon_labels(cfg)
    horizon_map = get_horizon_map(cfg)
    quantiles = cfg["training"]["quantiles"]

    # Initialize ensemble predictions accumulator: horizon -> quantile -> sum_weighted_preds
    ensemble_preds = {label: {q: 0.0 for q in quantiles} for label in horizon_labels}
    
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    total_weight = sum(weights)

    for m_path, weight in zip(model_paths, weights):
        model = joblib.load(m_path)
        if hasattr(model, "lookback"):
            lookback = model.lookback
            if len(x) < lookback:
                continue 
            context = x[:-1]
            m_preds = model.predict(x_last, context=context)
        else:
            try:
                m_preds = model.predict(x_last)
            except TypeError:
                m_preds = model.predict(len(x_last))
        
        for label in horizon_labels:
            for q in quantiles:
                ensemble_preds[label][q] += m_preds[label][q][-1] * (weight / total_weight)

    rows = []
    for label in horizon_labels:
        # Final pass: Ensure no quantile crossing by sorting the ensemble results
        q_vals = [ensemble_preds[label][q] for q in quantiles]
        q_vals.sort()
        sorted_preds = dict(zip(sorted(quantiles), q_vals))

        for q in quantiles:
            rows.append(
                {
                    "prediction_time": available["prediction_time"].iloc[-1],
                    "horizon": label,
                    "horizon_timedelta": str(horizon_map[label]),
                    "quantile": q,
                    "y_pred": float(sorted_preds[q]),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--asof", default=None)
    parser.add_argument("--output", default="forecast.csv")
    args = parser.parse_args()

    path = run_inference(args.config, args.model, args.asof, args.output)
    print(f"Saved forecast to {path}")
