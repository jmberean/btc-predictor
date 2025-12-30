from typing import Dict, List

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    naive_denom = np.mean(np.abs(np.diff(y_train)))
    if naive_denom == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_denom)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def balanced_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_up = y_true > 0
    y_pred_up = y_pred > 0
    tp = np.sum(y_true_up & y_pred_up)
    tn = np.sum(~y_true_up & ~y_pred_up)
    fp = np.sum(~y_true_up & y_pred_up)
    fn = np.sum(y_true_up & ~y_pred_up)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def reliability_curve(y_true: np.ndarray, quantile_preds: Dict[float, np.ndarray], bins: int = 10) -> pd.DataFrame:
    data = []
    for q, preds in sorted(quantile_preds.items()):
        coverage = np.mean(y_true <= preds)
        data.append({"quantile": q, "empirical_coverage": coverage})
    return pd.DataFrame(data)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
    y_train: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "mae": mae(y_true, y_pred_median),
        "rmse": rmse(y_true, y_pred_median),
        "smape": smape(y_true, y_pred_median),
        "mase": mase(y_true, y_pred_median, y_train),
        "directional_accuracy": directional_accuracy(y_true, y_pred_median),
        "balanced_directional_accuracy": balanced_directional_accuracy(y_true, y_pred_median),
    }
    for q, preds in quantile_preds.items():
        metrics[f"pinball_{q}"] = pinball_loss(y_true, preds, q)
    if 0.1 in quantile_preds and 0.9 in quantile_preds:
        metrics["interval_coverage_80"] = interval_coverage(
            y_true, quantile_preds[0.1], quantile_preds[0.9]
        )
    return metrics