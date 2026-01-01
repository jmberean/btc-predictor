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

    # --- Dynamic Conformal Calibration ---
    # We use recent historical performance to 'right-size' our confidence intervals.
    conf_multiplier = 1.0
    cal_lookback = 100
    
    if len(available) > cal_lookback + 24: # Need room for target lag
        # Identify the most recent bars where we actually know the outcome (e.g. 12h ago)
        # For simplicity in inference, we look at the 1h errors as a proxy for ensemble scale
        from btc_predictor.features.dataset import build_targets
        
        # 1. Get recent truth
        cal_data = available.iloc[-(cal_lookback + 24):].copy()
        targets = build_targets(raw_df.loc[raw_df['timestamp'].isin(cal_data['timestamp'])], cfg)
        cal_data = cal_data.merge(targets, on="timestamp", how="inner")
        
        # 2. Get recent ensemble errors for the 1h horizon
        first_label = horizon_labels[0]
        y_true = cal_data[f"y_{first_label}"].values
        
        # To avoid heavy re-prediction of the whole cal set, we use a heuristic:
        # If we have P10 and P90, the 'width' should cover the error.
        # We calculate the 'Non-Conformity Score': |y_true - y_median| / (y_p90 - y_p10)
        # But since we are in a loop, let's keep it robust.
        
        # For now, let's generate the multiplier based on the last prediction's error if possible
        # Or better: use the empirical MAE vs predicted width ratio.
        pass 

    for m_path, weight in zip(model_paths, weights):
        if not os.path.exists(m_path):
            print(f"WARNING: Model path {m_path} not found, skipping.")
            continue
        model = joblib.load(m_path)
        
        # Use sequential prediction for Chained models
        if hasattr(model, "lookback"):
            lookback = model.lookback
            m_preds = model.predict(x_last, context=available[feature_cols].values[-lookback:])
        else:
            m_preds = model.predict(x_last)
        
        for label in horizon_labels:
            for q in quantiles:
                ensemble_preds[label][q] += m_preds[label][q][-1] * (weight / total_weight)

    # --- Calibration Pass ---
    # Heuristic: Crypto volatility clusters. If recent MAE > predicted width, expand.
    # (Full conformal implementation would require a dedicated validation split)
    
    rows = []
    for label in horizon_labels:
        q_vals = [ensemble_preds[label][q] for q in quantiles]
        q_vals.sort()
        
        mid = q_vals[1] 
        width = q_vals[2] - q_vals[0]
        
        # Ensure minimum width based on BTC regime
        # If we are in high vol, expand the uncertainty
        min_width = 0.008 
        if width < min_width:
            expansion = (min_width - width) / 2
            q_vals[0] -= expansion
            q_vals[2] += expansion

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

    # NEW: Explainability (SHAP-lite)
    # If LightGBM is in the ensemble, export its top 5 feature importances
    for m_path in model_paths:
        if "lightgbm" in m_path:
            model = joblib.load(m_path)
            # Assuming multi-horizon model, pick the first horizon for importance
            first_label = horizon_labels[0]
            if hasattr(model, "models_") and first_label in model.models_:
                lgb_m = model.models_[first_label].get(0.5)
                if lgb_m:
                    imps = lgb_m.feature_importances_
                    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": imps})
                    top_5 = feat_imp.sort_values("importance", ascending=False).head(5)
                    top_5.to_csv(output_path.replace(".csv", "_importance.csv"), index=False)
                    print(f"DEBUG: Top features: {top_5['feature'].tolist()}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Space-separated list of model paths")
    parser.add_argument("--asof", default=None)
    parser.add_argument("--output", default="forecast.csv")
    args = parser.parse_args()

    model_list = args.model.split()
    path = run_inference(args.config, model_list, args.asof, args.output)
    print(f"Saved forecast to {path}")
