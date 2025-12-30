from typing import Dict

import numpy as np
import pandas as pd


def backtest_simple(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    threshold: float,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, float]:
    signal = np.where(np.abs(y_pred_median) >= threshold, np.sign(y_pred_median), 0)
    signal = pd.Series(signal).shift(1).fillna(0).values
    costs = (fee_bps + slippage_bps) / 10000.0
    turnover = np.abs(np.diff(signal, prepend=0))
    net_returns = signal * y_true - turnover * costs

    if len(net_returns) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0}

    ann_factor = np.sqrt(24 * 365)
    sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * ann_factor
    equity = np.cumsum(net_returns)
    drawdown = equity - np.maximum.accumulate(equity)
    max_drawdown = drawdown.min()
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover": float(np.mean(turnover)),
    }