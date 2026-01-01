import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from btc_predictor.evaluation.metrics import reliability_curve
from btc_predictor.evaluation.plots import plot_forecast_intervals, plot_reliability
from btc_predictor.utils.logging import setup_logging


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in metrics_df.columns if c not in {"model", "fold", "horizon", "horizon_timedelta", "n_test"}]
    return metrics_df.groupby(["model", "horizon"])[metric_cols].mean().reset_index()


def stability_by_regime(preds_df: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    preds_df = preds_df.sort_values("timestamp")
    preds_df["rolling_vol"] = preds_df.groupby(["model", "horizon", "quantile"])["y_true"].transform(
        lambda x: x.rolling(window).std()
    )
    preds_df["vol_regime"] = preds_df["rolling_vol"] > preds_df["rolling_vol"].median()

    summary = (
        preds_df.groupby(["model", "horizon", "vol_regime"])
        .apply(lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"])))
        .reset_index(name="mae")
    )
    return summary


def performance_by_period(preds_df: pd.DataFrame) -> pd.DataFrame:
    df = preds_df[preds_df["quantile"] == 0.5].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = df["timestamp"].dt.year
    df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)
    yearly = (
        df.groupby(["model", "horizon", "year"])
        .apply(lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"])))
        .reset_index(name="mae")
    )
    quarterly = (
        df.groupby(["model", "horizon", "quarter"])
        .apply(lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"])))
        .reset_index(name="mae")
    )
    yearly["period_type"] = "year"
    quarterly["period_type"] = "quarter"
    combined = pd.concat(
        [
            yearly.rename(columns={"year": "period"}),
            quarterly.rename(columns={"quarter": "period"}),
        ],
        ignore_index=True,
    )
    return combined


def main():
    setup_logging("evaluate")
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()

    metrics_path = os.path.join(args.artifacts, "metrics.csv")
    preds_path = os.path.join(args.artifacts, "predictions.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
        raise FileNotFoundError("metrics.csv and predictions.csv are required")

    print(f"Loading metrics from {metrics_path}...")
    metrics_df = pd.read_csv(metrics_path)
    print(f"Loading predictions from {preds_path}...")
    preds_df = pd.read_csv(preds_path)

    print("Aggregating metrics...")
    agg = aggregate_metrics(metrics_df)
    agg.to_csv(os.path.join(args.artifacts, "metrics_aggregated.csv"), index=False)

    plots_dir = os.path.join(args.artifacts, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating plots in {plots_dir}...")
    groups = list(preds_df.groupby(["model", "horizon"]))
    total_groups = len(groups)
    
    for i, ((model, horizon), group) in enumerate(groups, 1):
        print(f"  [{i}/{total_groups}] Plotting {model} - {horizon}...", end="\r")
        pivot = group.pivot_table(index=["timestamp", "prediction_time", "y_true"], columns="quantile", values="y_pred")
        pivot = pivot.reset_index()
        quantile_preds = {q: pivot[q].values for q in pivot.columns if isinstance(q, float)}
        if not quantile_preds:
            continue
        plot_forecast_intervals(
            pivot["timestamp"],
            pivot["y_true"].values,
            quantile_preds,
            title=f"{model} {horizon}",
            path=os.path.join(plots_dir, f"forecast_{model}_{horizon}.png"),
        )
        rel = reliability_curve(pivot["y_true"].values, quantile_preds)
        plot_reliability(
            rel,
            title=f"Reliability {model} {horizon}",
            path=os.path.join(plots_dir, f"reliability_{model}_{horizon}.png"),
        )
    print("\nPlotting complete.")

    print("Calculating stability by regime...")
    stability = stability_by_regime(preds_df)
    stability.to_csv(os.path.join(args.artifacts, "stability_by_regime.csv"), index=False)

    print("Calculating performance by period...")
    periodic = performance_by_period(preds_df)
    periodic.to_csv(os.path.join(args.artifacts, "performance_by_period.csv"), index=False)
    
    print("Evaluation complete.")
