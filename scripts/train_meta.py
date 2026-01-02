import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from btc_predictor.config import load_config
from btc_predictor.features.engineering import build_feature_frame, feature_columns
from btc_predictor.data.binance_bulk import load_binance_bulk_klines
from btc_predictor.data.external import load_external_features
from btc_predictor.features.dataset import build_supervised_dataset
from btc_predictor.trading import backtest_simple

def train_meta_model(artifacts_dir):
    print(f"--- Training Meta-Model for {artifacts_dir} ---")
    
    # 1. Load Config & Predictions
    cfg_path = os.path.join(artifacts_dir, "config.json")
    preds_path = os.path.join(artifacts_dir, "predictions.csv")
    
    if not os.path.exists(cfg_path):
        print("Config not found.")
        return
    if not os.path.exists(preds_path):
        print("Predictions not found.")
        return
        
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
        
    preds_df = pd.read_csv(preds_path)
    # Filter for LightGBM 1h Median
    preds_df = preds_df[(preds_df["model"] == "lightgbm") & 
                        (preds_df["horizon"] == "1h") & 
                        (preds_df["quantile"] == 0.5)].copy()
    
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])
    preds_df["prediction_time"] = pd.to_datetime(preds_df["prediction_time"]).dt.tz_localize("UTC")
    preds_df = preds_df.sort_values("timestamp")
    
    # 2. Re-create Features (to give context to the Meta Model)
    print("Re-building feature set...")
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
    
    # Merge Features with Predictions
    # We match on prediction_time (available_at)
    meta_df = pd.merge(preds_df, feature_df, on="prediction_time", how="inner")
    
    # 3. Define Meta-Target
    # Target: Did the trade make money? (considering direction and fee)
    fee = 0.0005
    # If model said UP, price must go UP > fee.
    # If model said DOWN, price must go DOWN > fee.
    # If model said NOTHING (pred < threshold), it's ignored (or Class 0).
    
    # Let's focus on filtering ACTIVE signals.
    threshold = 0.0005
    meta_df["is_signal"] = np.abs(meta_df["y_pred"]) > threshold
    meta_df = meta_df[meta_df["is_signal"]].copy()
    
    # Success = Direction Correct AND Magnitude > Fee
    # Actually, simpler: Is the Net Return > 0?
    direction = np.sign(meta_df["y_pred"])
    meta_df["net_return"] = direction * meta_df["y_true"] - fee
    meta_df["meta_target"] = (meta_df["net_return"] > 0).astype(int)
    
    print(f"Meta-Dataset: {len(meta_df)} active trade signals.")
    print(f"Baseline Win Rate: {meta_df['meta_target'].mean():.2%}")
    
    # 4. Train/Test Split (Chronological)
    split_idx = int(len(meta_df) * 0.8)
    train_df = meta_df.iloc[:split_idx]
    test_df = meta_df.iloc[split_idx:]
    
    # 5. Features for Meta-Model
    # We want features that indicate "Regime"
    meta_features = [
        "rsi", "atr", "vol_regime", "trend_regime", 
        "volume_z_24", "hour", "dow", 
        "oi_z", "ls_ratio_z",  # High-Alpha features
        "y_pred" # The confidence of the primary model
    ]
    # Ensure they exist
    meta_features = [f for f in meta_features if f in meta_df.columns]
    
    X_train = train_df[meta_features].fillna(0)
    y_train = train_df["meta_target"]
    X_test = test_df[meta_features].fillna(0)
    y_test = test_df["meta_target"]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    meta_preds = clf.predict(X_test)
    meta_probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, meta_preds)
    prec = precision_score(y_test, meta_preds)
    print(f"Meta-Model Accuracy: {acc:.2%}")
    print(f"Meta-Model Precision: {prec:.2%} (This is the new Win Rate)")
    
    # 7. Simulated Equity Curve
    # Baseline Strategy (Take all signals)
    base_pnl = test_df["net_return"].sum()
    base_sharpe = test_df["net_return"].mean() / test_df["net_return"].std() * np.sqrt(24*365)
    
    # Meta Strategy (Take only if Prob > 0.55)
    filter_mask = meta_probs > 0.55
    meta_pnl = test_df.loc[test_df.index[filter_mask], "net_return"].sum()
    
    filtered_returns = test_df.loc[test_df.index[filter_mask], "net_return"]
    if len(filtered_returns) > 0:
        meta_sharpe = filtered_returns.mean() / filtered_returns.std() * np.sqrt(24*365)
    else:
        meta_sharpe = 0
    
    print("\n--- Impact Analysis (Hold-out Set) ---")
    print(f"Original PnL: {base_pnl:.4f} (Sharpe: {base_sharpe:.2f})")
    print(f"Filtered PnL: {meta_pnl:.4f} (Sharpe: {meta_sharpe:.2f})")
    print(f"Trades Taken: {sum(filter_mask)} / {len(test_df)}")

    # Save
    model_path = os.path.join(artifacts_dir, "models", "meta_rf.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved Meta-Model to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    train_meta_model(args.artifacts)
