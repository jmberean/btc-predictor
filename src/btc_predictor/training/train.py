import json
import os
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from btc_predictor.config import get_horizon_labels, get_horizons, load_config
from btc_predictor.data.binance_bulk import load_binance_bulk_klines
from btc_predictor.data.external import load_external_features
from btc_predictor.data.ingestion import fetch_ohlcv_ccxt, load_ohlcv_csv
from btc_predictor.evaluation.metrics import compute_metrics
from btc_predictor.evaluation.walk_forward import generate_walk_forward_splits
from btc_predictor.features.availability import assert_point_in_time
from btc_predictor.features.dataset import build_supervised_dataset, split_xy
from btc_predictor.features.engineering import build_feature_frame, feature_columns
from btc_predictor.models.registry import init_model
from btc_predictor.utils.seed import set_seed


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


def _model_fit_predict(model_name: str, model, x_train, y_train, x_test):
    if model_name in {"naive", "random_walk", "arima", "garch"}:
        model.fit(y_train)
        preds = model.predict(len(x_test))
    elif model_name in {"lstm", "nbeats"}:
        model.fit(x_train, y_train)
        preds = model.predict(x_test, context=x_train)
    else:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
    return preds


def _tune_lightgbm_params(x_train, y_train, cfg, seed: int):
    from btc_predictor.models.tree import LightGBMQuantileModel

    base_params = dict(cfg.get("lightgbm", {}))
    base_params.setdefault("random_state", seed)
    max_trials = int(cfg["training"].get("max_trials", 10))
    if max_trials <= 1:
        return base_params

    horizon_labels = get_horizon_labels(cfg)
    short_label = horizon_labels[0]
    n = len(x_train)
    if n < 100:
        return base_params
    split_idx = int(n * 0.8)
    x_sub, x_val = x_train[:split_idx], x_train[split_idx:]
    y_sub, y_val = y_train[short_label][:split_idx], y_train[short_label][split_idx:]

    search_space = {
        "learning_rate": [0.01, 0.03, 0.05],
        "num_leaves": [31, 63, 127],
        "min_data_in_leaf": [20, 50, 100],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "n_estimators": [200, 400, 600],
    }

    rng = np.random.RandomState(seed)
    sampler = ParameterSampler(search_space, n_iter=min(max_trials, 20), random_state=rng)
    best_params = base_params
    best_mae = float("inf")

    for params in sampler:
        trial_params = dict(base_params)
        trial_params.update(params)
        model = LightGBMQuantileModel(params=trial_params, quantiles=[0.5], horizons=[short_label])
        model.fit(x_sub, {short_label: y_sub})
        pred = model.predict(x_val)[short_label][0.5]
        err = np.mean(np.abs(y_val - pred))
        if err < best_mae:
            best_mae = err
            best_params = trial_params

    return best_params


def run_train(cfg_path: str) -> str:
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))

    raw_df = _load_data(cfg)
    external_dfs = load_external_features(cfg)
    feature_df = build_feature_frame(raw_df, cfg, external_dfs=external_dfs)
    dataset = build_supervised_dataset(raw_df, feature_df, cfg)
    dataset = dataset.sort_values("prediction_time").reset_index(drop=True)
    assert_point_in_time(dataset)

    horizons = get_horizons(cfg)
    horizon_labels = get_horizon_labels(cfg)
    feature_cols = feature_columns(dataset)
    dataset = dataset.dropna(subset=feature_cols).reset_index(drop=True)
    splits = generate_walk_forward_splits(dataset, cfg)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("artifacts", run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    all_metrics = []
    all_predictions = []
    all_importances = []

    for model_name in cfg["training"]["models"]:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = dataset.loc[train_idx]
            test_df = dataset.loc[test_idx]

            x_train, y_train = split_xy(train_df, horizon_labels, feature_cols)
            x_test, y_test = split_xy(test_df, horizon_labels, feature_cols)

            if model_name == "lightgbm":
                from btc_predictor.models.tree import LightGBMQuantileModel

                tuned_params = _tune_lightgbm_params(x_train.values, y_train, cfg, seed=cfg.get("seed", 42))
                model = LightGBMQuantileModel(
                    params=tuned_params,
                    quantiles=cfg["training"]["quantiles"],
                    horizons=horizon_labels,
                )
            else:
                model = init_model(model_name, cfg, input_size=len(feature_cols))
            preds = _model_fit_predict(model_name, model, x_train.values, y_train, x_test.values)

            for label, h in zip(horizon_labels, horizons):
                quantile_preds = preds[label]
                median_pred = quantile_preds.get(0.5, list(quantile_preds.values())[0])
                metrics = compute_metrics(
                    y_true=y_test[label],
                    y_pred_median=np.asarray(median_pred),
                    quantile_preds={q: np.asarray(p) for q, p in quantile_preds.items()},
                    y_train=y_train[label],
                )
                metrics.update(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "horizon": label,
                        "horizon_timedelta": str(h),
                        "n_test": len(test_df),
                    }
                )
                all_metrics.append(metrics)

                for q, pred_vals in quantile_preds.items():
                    all_predictions.append(
                        pd.DataFrame(
                            {
                                "model": model_name,
                                "fold": fold_idx,
                                "horizon": label,
                                "horizon_timedelta": str(h),
                                "timestamp": test_df["timestamp"].values,
                                "prediction_time": test_df["prediction_time"].values,
                                "quantile": q,
                                "y_true": y_test[label],
                                "y_pred": pred_vals,
                            }
                        )
                    )

            if model_name == "lightgbm":
                for label, h in zip(horizon_labels, horizons):
                    model_q = model.models_[label].get(0.5)
                    if model_q is None:
                        continue
                    importances = model_q.feature_importances_
                    all_importances.append(
                        pd.DataFrame(
                            {
                                "model": model_name,
                                "fold": fold_idx,
                                "horizon": label,
                                "horizon_timedelta": str(h),
                                "feature": feature_cols,
                                "importance": importances,
                            }
                        )
                    )

            model_path = os.path.join(out_dir, "models", f"{model_name}_fold{fold_idx}.joblib")
            try:
                joblib.dump(model, model_path)
            except Exception:
                # Some models may not be serializable; skip if needed.
                pass

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    if all_predictions:
        preds_df = pd.concat(all_predictions, ignore_index=True)
        preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    if all_importances:
        imp_df = pd.concat(all_importances, ignore_index=True)
        imp_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = run_train(args.config)
    print(f"Saved artifacts to {output}")
