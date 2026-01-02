
import os
import sys
import pandas as pd
import joblib
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from btc_predictor.config import load_config, get_horizon_labels
from btc_predictor.features.engineering import build_feature_frame, feature_columns
from btc_predictor.data.external import load_external_features
from btc_predictor.data.binance_bulk import load_binance_bulk_klines
from btc_predictor.evaluation.walk_forward import generate_walk_forward_splits
from btc_predictor.features.dataset import build_supervised_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()

    # 1. Load Config
    cfg_path = os.path.join(args.artifacts, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        print("Config.json missing, falling back to configs/binance_bulk.yaml")
        from btc_predictor.config import load_config
        cfg = load_config("configs/binance_bulk.yaml")

    # 2. Load Data
    print("Loading data...")
    # Re-implement data loading logic briefly
    bulk_cfg = cfg["data"]["binance_bulk"]
    raw_df = load_binance_bulk_klines(
        bulk_cfg["klines_path"],
        timeframe=bulk_cfg.get("timeframe", cfg["data"]["timeframe"]),
        tz=cfg["data"].get("timezone", "UTC"),
        start=cfg["data"].get("start"),
        end=cfg["data"].get("end"),
    )
    external_dfs = load_external_features(cfg)
    feature_df = build_feature_frame(raw_df, cfg, external_dfs=external_dfs)
    dataset = build_supervised_dataset(raw_df, feature_df, cfg)
    dataset = dataset.sort_values("prediction_time").reset_index(drop=True)
    
    feature_cols = feature_columns(dataset)
    dataset = dataset.dropna(subset=feature_cols).reset_index(drop=True)
    splits = generate_walk_forward_splits(dataset, cfg)
    
    x_all = dataset[feature_cols].to_numpy()
    horizon_labels = get_horizon_labels(cfg)
    y_all = {label: dataset[f"y_{label}"].to_numpy() for label in horizon_labels}
    
    all_predictions = []
    
    # 3. Iterate Models and Folds
    model_name = "lightgbm" # Only recover LightGBM for now
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        model_path = os.path.join(args.artifacts, "models", f"{model_name}_fold{fold_idx}.joblib")
        if not os.path.exists(model_path):
            print(f"Skipping missing model: {model_path}")
            continue
            
        print(f"Generating preds for {model_name} Fold {fold_idx}...")
        model = joblib.load(model_path)
        
        test_df = dataset.loc[test_idx]
        x_test = x_all[test_idx]
        y_test = {label: y_all[label][test_idx] for label in horizon_labels}
        
        preds = model.predict(x_test)
        
        for label in horizon_labels:
            quantile_preds = preds[label]
            for q, pred_vals in quantile_preds.items():
                all_predictions.append(pd.DataFrame({
                    "model": model_name,
                    "fold": fold_idx,
                    "horizon": label,
                    "timestamp": test_df["timestamp"].values,
                    "prediction_time": test_df["prediction_time"].values,
                    "quantile": q,
                    "y_true": y_test[label],
                    "y_pred": pred_vals
                }))

    # 4. Save
    if all_predictions:
        out_df = pd.concat(all_predictions, ignore_index=True)
        out_path = os.path.join(args.artifacts, "predictions.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()
