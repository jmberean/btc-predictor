from typing import Dict, Tuple

import numpy as np
import pandas as pd


def backtest_simple(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    threshold: float,
    fee_bps: float,
    slippage_bps: float,
    max_leverage: float = 1.0,
    stop_loss_pct: float = 0.03, # 3% Hard Stop
) -> Tuple[Dict[str, float], pd.DataFrame]:
    # 1. Generate Raw Signals
    raw_signal = np.where(np.abs(y_pred_median) >= threshold, np.sign(y_pred_median), 0)
    
    # 2. Dynamic Position Sizing (Volatility Targeting)
    # We use a rolling window of the true returns to estimate current risk
    window = 168 # 1 week
    rolling_vol = pd.Series(y_true).rolling(window, min_periods=24).std().fillna(method='bfill').values
    # Target a constant 1% risk per trade (standardized by vol)
    target_vol = 0.01 
    pos_size = np.clip(target_vol / (rolling_vol + 1e-8), 0, max_leverage)
    
    # 3. Apply Signal + Size (Shifted by 1 to prevent leakage)
    signal = pd.Series(raw_signal * pos_size).shift(1).fillna(0).values
    
    # 4. Calculate Costs
    costs = (fee_bps + slippage_bps) / 10000.0
    turnover = np.abs(np.diff(signal, prepend=0))
    
    # 5. Apply Stop-Loss Logic (Simulated)
    # If the raw return is worse than our stop_loss, we truncate the loss
    # (Simplified: assumes we exit at exactly the stop loss level)
    trade_returns = signal * y_true
    sl_mask = (signal == 1) & (y_true < -stop_loss_pct) | (signal == -1) & (y_true > stop_loss_pct)
    trade_returns[sl_mask] = -stop_loss_pct * np.abs(signal[sl_mask])
    
    net_returns = trade_returns - (turnover * costs)

    history = pd.DataFrame({
        "timestamp": timestamps.values,
        "signal": signal,
        "raw_return": y_true,
        "net_return": net_returns,
        "turnover": turnover,
        "equity": np.cumsum(net_returns)
    })

    if len(net_returns) == 0 or np.std(net_returns) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0}, history

    ann_factor = np.sqrt(24 * 365)
    sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * ann_factor
    equity = history["equity"].values
    drawdown = equity - np.maximum.accumulate(equity)
    max_drawdown = drawdown.min()
    
    metrics = {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover": float(np.mean(turnover)),
    }
    
    return metrics, history