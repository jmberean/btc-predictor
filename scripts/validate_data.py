import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from btc_predictor.config import load_config
from btc_predictor.training.train import _load_data
from btc_predictor.data.external import load_external_features
from btc_predictor.features.engineering import build_feature_frame

def validate_data(config_path):
    print(f"--- Validating Data Pipeline with config: {config_path} ---")
    cfg = load_config(config_path)
    
    # 1. Load Base Data (Spot)
    print("\n1. Loading Base Spot Data...")
    raw_df = _load_data(cfg)
    print(f"   Shape: {raw_df.shape}")
    print(f"   Range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
    
    # 2. Load External Data
    print("\n2. Loading External Data (Funding, Metrics, etc.)...")
    external_dfs = load_external_features(cfg)
    print(f"   Loaded {len(external_dfs)} external dataframes.")
    
    for i, df in enumerate(external_dfs):
        cols = list(df.columns)
        # Identify what kind of data this is based on columns
        label = "Unknown"
        if "funding_rate" in cols: label = "Funding Rate"
        elif "open_interest" in cols: label = "Metrics (OI)"
        elif "mark_close" in cols: label = "Mark Price"
        elif "index_close" in cols: label = "Index Price"
        
        print(f"   [Ext {i}] {label}: Shape {df.shape}, Range {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"       Columns: {cols}")
        if "available_at" in cols:
             delay = (df["available_at"] - df["timestamp"]).mean()
             print(f"       Avg Delay: {delay}")

    # 3. Build Feature Frame
    print("\n3. Building Feature Frame (Merging)...")
    try:
        feature_df = build_feature_frame(raw_df, cfg, external_dfs=external_dfs)
    except Exception as e:
        print(f"   ERROR building feature frame: {e}")
        return

    print(f"   Final Shape: {feature_df.shape}")
    print(f"   Columns: {list(feature_df.columns)}")
    
    # 4. Check for New Features
    print("\n4. Checking Key Feature Coverage:")
    key_features = [
        "funding_rate", 
        "open_interest", 
        "ls_ratio", 
        "taker_ls_ratio",
        "oi_chg_1", 
        "oi_z",
        "ls_ratio_z", 
        "basis_mark"
    ]
    
    for feat in key_features:
        if feat in feature_df.columns:
            non_null = feature_df[feat].count()
            total = len(feature_df)
            pct = (non_null / total) * 100
            print(f"   - {feat}: {non_null}/{total} ({pct:.1f}%)")
            # Show last few values
            last_val = feature_df[feat].iloc[-1]
            print(f"     Last Value ({feature_df['timestamp'].iloc[-1]}): {last_val}")
        else:
            print(f"   - {feat}: [MISSING]")

    # 5. Check Alignment
    print("\n5. Data Alignment Check (Tail):")
    cols_to_show = ["timestamp", "close"] + [k for k in key_features if k in feature_df.columns]
    print(feature_df[cols_to_show].tail(10))

if __name__ == "__main__":
    validate_data("configs/binance_bulk.yaml")
