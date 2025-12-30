import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from btc_predictor.trading import backtest_simple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--horizon", default="1h")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    preds_path = os.path.join(args.artifacts, "predictions.csv")
    if not os.path.exists(preds_path):
        raise FileNotFoundError("predictions.csv is required")

    preds_df = pd.read_csv(preds_path)
    df = preds_df[(preds_df["quantile"] == 0.5) & (preds_df["horizon"] == args.horizon)].copy()
    if args.model:
        df = df[df["model"] == args.model]

    if df.empty:
        raise ValueError("No predictions found for the requested filters")

    results = []
    for model, group in df.groupby("model"):
        metrics = backtest_simple(
            timestamps=group["timestamp"],
            y_true=group["y_true"].values,
            y_pred_median=group["y_pred"].values,
            threshold=0.0005,
            fee_bps=5,
            slippage_bps=2,
        )
        metrics.update({"model": model, "horizon": args.horizon})
        results.append(metrics)

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(args.artifacts, "trading_metrics.csv"), index=False)
    print(out_df)


if __name__ == "__main__":
    main()