from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_forecast_intervals(timestamps, y_true, quantile_preds: Dict[float, np.ndarray], title: str, path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, y_true, label="actual", color="black", linewidth=1)
    if 0.5 in quantile_preds:
        plt.plot(timestamps, quantile_preds[0.5], label="median", color="blue")
    if 0.1 in quantile_preds and 0.9 in quantile_preds:
        plt.fill_between(
            timestamps,
            quantile_preds[0.1],
            quantile_preds[0.9],
            color="lightblue",
            alpha=0.4,
            label="P10-P90",
        )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reliability(reliability_df, title: str, path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(reliability_df["quantile"], reliability_df["empirical_coverage"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical coverage")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()