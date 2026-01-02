
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from btc_predictor.trading import backtest_simple

def max_drawdown(equity_curve):
    if len(equity_curve) == 0: return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max)
    # Return max drawdown as a negative number
    return drawdown.min()

def evaluate_barrier(artifacts_dir):
    print(f"--- Evaluating Triple Barrier Model: {artifacts_dir} ---")
    
    preds_path = os.path.join(artifacts_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        print("predictions.csv not found")
        return
        
    df = pd.read_csv(preds_path)
    # y_pred is the probability of winning (Class 2)
    # y_true is the label (1 for Win, -1 for Loss, 0 for Timeout)
    
    # We need to simulate PnL.
    # Win = +2%
    # Loss = -1%
    # Timeout = 0% (Conservative)
    
    PROFIT_TARGET = 0.02
    STOP_LOSS = 0.01
    FEE = 0.0005 * 2 # Entry + Exit
    
    # Grid search for best probability threshold
    best_sharpe = -100
    best_thresh = 0
    
    print("\nThreshold Scan:")
    for thresh in np.arange(0.3, 0.8, 0.05):
        # Signal: Prob > Threshold
        signal = df["y_pred"] > thresh
        
        # Calculate Returns per trade
        # Vectorized logic
        returns = np.zeros(len(df))
        
        # If Signal=True:
        #   If y_true=1: +Profit
        #   If y_true=-1: -Stop
        #   If y_true=0: 0
        
        # Apply logic only where signal is True
        win_mask = (df["y_true"] == 1) & signal
        loss_mask = (df["y_true"] == -1) & signal
        timeout_mask = (df["y_true"] == 0) & signal
        
        returns[win_mask] = PROFIT_TARGET - FEE
        returns[loss_mask] = -STOP_LOSS - FEE
        returns[timeout_mask] = 0.0 - FEE # Timeout still pays fees to close
        
        # If no trades, skip
        if np.sum(signal) < 10:
            continue
            
        # Equity Curve
        equity = np.cumsum(returns)
        
        # Sharpe
        if np.std(returns) == 0: continue
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24*365/12) # Annualized (12h steps approx) 
        mdd = max_drawdown(equity)
        win_rate = np.mean(returns[signal] > 0)
        
        print(f"Thresh {thresh:.2f} | Trades: {np.sum(signal)} | Sharpe: {sharpe:.2f} | MDD: {mdd:.2%} | Win Rate: {win_rate:.1%}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thresh = thresh

    print(f"\nBest Config: Threshold {best_thresh:.2f} -> Sharpe {best_sharpe:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    evaluate_barrier(args.artifacts)
