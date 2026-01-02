
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from btc_predictor.trading import backtest_simple

def run_ensemble_backtest(artifacts_dir):
    print(f"--- Running Ensemble Backtest on {artifacts_dir} ---")
    
    # 1. Load Predictions
    preds_path = os.path.join(artifacts_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        print("Error: predictions.csv not found.")
        return

    df = pd.read_csv(preds_path)
    
    # 2. Filter for 1h Horizon and Median Quantile (0.5)
    # We focus on 1h first as it was the most profitable
    horizon = "1h"
    df = df[(df["horizon"] == horizon) & (df["quantile"] == 0.5)].copy()
    
    # 3. Pivot to get columns: [timestamp, y_true, lightgbm_pred, lstm_pred, ...]
    pivoted = df.pivot(index="timestamp", columns="model", values="y_pred")
    
    # Add True Target (y_true is the same for all models, just take first)
    y_true = df.groupby("timestamp")["y_true"].first()
    pivoted["y_true"] = y_true
    
    # 4. Construct Ensembles
    # A. All Models
    all_models = [c for c in pivoted.columns if c != "y_true"]
    pivoted["ensemble_all"] = pivoted[all_models].mean(axis=1)
    
    # B. The "Smart" Ensemble (LightGBM + N-BEATS, ignoring the weak LSTM/Naive/Arima)
    smart_models = ["lightgbm", "nbeats"]
    # Check if they exist in this run
    available_smart = [m for m in smart_models if m in pivoted.columns]
    if available_smart:
        pivoted["ensemble_smart"] = pivoted[available_smart].mean(axis=1)
    
    # 5. Simulate Trading
    fee = 0.0005
    results = []
    
    strategies = ["ensemble_all", "ensemble_smart", "lightgbm"] # Compare against best single
    
    for strat in strategies:
        if strat not in pivoted.columns: continue
        
        metrics, history = backtest_simple(
            timestamps=pivoted.index,
            y_true=pivoted["y_true"].values,
            y_pred_median=pivoted[strat].values,
            threshold=fee,
            fee_bps=5,
            slippage_bps=2
        )
        
        # Calculate win rate for non-zero signals
        trades = history[history["signal"] != 0]
        win_rate = (trades["net_return"] > 0).mean() if not trades.empty else 0
        
        results.append({
            "Strategy": strat,
            "Sharpe": metrics["sharpe"],
            "MDD": metrics["max_drawdown"],
            "Win Rate": win_rate
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Ensemble Backtest Results (1h Horizon) ---")
    print(res_df.sort_values("Sharpe", ascending=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    run_ensemble_backtest(args.artifacts)
