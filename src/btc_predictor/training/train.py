import json
import os
from datetime import datetime
from typing import Dict

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
from btc_predictor.features.dataset import build_supervised_dataset
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
    base_params.setdefault("n_jobs", cfg["training"].get("n_jobs", -1))
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
    x_all = dataset[feature_cols].to_numpy()
    y_all = {label: dataset[f"y_{label}"].to_numpy() for label in horizon_labels}

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("artifacts", run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    all_metrics = []
    all_predictions = []
    all_importances = []

    for model_name in cfg["training"]["models"]:
        print(f"--- Training model: {model_name} ---")
        # Pre-tune LightGBM once if needed
        tuned_params = {}
        if model_name == "lightgbm":
             # Use the first split's training data for tuning to avoid leakage
             # Or use a dedicated subset. Here we use the first fold's train set.
             if len(splits) > 0:
                 print("Tuning LightGBM parameters...")
                 tune_train_idx = splits[0][0]
                 tune_x = x_all[tune_train_idx]
                 tune_y = {label: y_all[label][tune_train_idx] for label in horizon_labels}
                 tuned_params = _tune_lightgbm_params(tune_x, tune_y, cfg, seed=cfg.get("seed", 42))
                 print(f"Best params found: {tuned_params}")
             else:
                 # Fallback if no splits
                 tuned_params = dict(cfg.get("lightgbm", {}))

        num_folds = len(splits)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"Fold {fold_idx + 1}/{num_folds}...")
            test_df = dataset.loc[test_idx]
            x_train_values = x_all[train_idx]
            x_test_values = x_all[test_idx]
            y_train = {label: y_all[label][train_idx] for label in horizon_labels}
            y_test = {label: y_all[label][test_idx] for label in horizon_labels}

            if model_name == "lightgbm":
                from btc_predictor.models.tree import LightGBMQuantileModel

                model = LightGBMQuantileModel(
                    params=tuned_params,
                    quantiles=cfg["training"]["quantiles"],
                    horizons=horizon_labels,
                )
                early_stopping = cfg["training"].get("early_stopping_rounds")
                val_fraction = cfg.get("lightgbm", {}).get("val_fraction", cfg["training"].get("val_fraction", 0.2))
                min_val_size = cfg.get("lightgbm", {}).get("min_val_size", 100)
                x_sub = x_train_values
                y_sub = y_train
                x_val = None
                y_val = None
                if early_stopping:
                    val_size = int(len(x_train_values) * val_fraction)
                    if val_size >= min_val_size and len(x_train_values) - val_size >= 1:
                        x_sub = x_train_values[:-val_size]
                        x_val = x_train_values[-val_size:]
                        y_sub = {label: y_train[label][:-val_size] for label in horizon_labels}
                        y_val = {label: y_train[label][-val_size:] for label in horizon_labels}
                model.fit(
                    x_sub,
                    y_sub,
                    x_val=x_val,
                    y_val=y_val,
                    early_stopping_rounds=early_stopping,
                    refit_full=bool(x_val is not None),
                    x_full=x_train_values,
                    y_full=y_train,
                )
                preds = model.predict(x_test_values)
            else:
                model = init_model(model_name, cfg, input_size=len(feature_cols))
                preds = _model_fit_predict(model_name, model, x_train_values, y_train, x_test_values)

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
            print(f"Fold {fold_idx + 1} completed.")

            model_path = os.path.join(out_dir, "models", f"{model_name}_fold{fold_idx}.joblib")
            try:
                # If model has a 'net' attribute (PyTorch), try to save it specifically or ensure joblib can handle it
                # LSTMQuantileModel and NBEATSModel both have .net
                joblib.dump(model, model_path)
            except Exception as e:
                print(f"WARNING: Could not save model {model_name} fold {fold_idx}: {e}")
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
