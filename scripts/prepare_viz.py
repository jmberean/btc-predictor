import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path

def load_all_hist():
    monthly_files = sorted(glob.glob("data/binance/data/spot/monthly/klines/BTCUSDT/1h/*.csv"))
    daily_files = sorted(glob.glob("data/binance/data/spot/daily/klines/BTCUSDT/1h/*.csv"))
    all_files = monthly_files + daily_files
    frames = []
    for f in all_files:
        df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4], names=["time", "open", "high", "low", "close"])
        frames.append(df)
    full_df = pd.concat(frames).tail(200) # Need more for rolling vol
    sample_ts = full_df["time"].iloc[0]
    unit = 'us' if sample_ts > 1e14 else 'ms'
    full_df["time_dt"] = pd.to_datetime(full_df["time"], unit=unit, utc=True)
    return full_df

def main():
    # Load history
    hist_df = load_all_hist()
    # Calculate current rolling volatility (same as train config: 168h)
    log_ret = np.log(hist_df["close"] / hist_df["close"].shift(1))
    current_vol = log_ret.tail(168).std()

    last_price = hist_df["close"].iloc[-1]
    last_time = int(hist_df["time_dt"].iloc[-1].timestamp()) 

    hist_json = []
    for row in hist_df.tail(100).itertuples():
        hist_json.append({
            "time": int(row.time_dt.timestamp()),
            "open": row.open, "high": row.high, "low": row.low, "close": row.close
        })

    # Load 10/10 ensemble forecast (Z-scores) from forecasts/ directory
    forecast_df = pd.read_csv("forecasts/latest_ensemble.csv")

    forecast_viz = []
    forecast_viz.append({"time": last_time, "p10": last_price, "p50": last_price, "p90": last_price})

    for horizon in ["1h", "2h", "4h", "8h", "12h", "24h"]:
        h_int = int(horizon.replace("h", ""))
        target_time = last_time + (h_int * 3600)
        scale = current_vol * np.sqrt(h_int)
        
        z10 = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.1)]["y_pred"].values[0]
        z50 = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.5)]["y_pred"].values[0]
        z90 = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.9)]["y_pred"].values[0]
        
        sorted_rets = sorted([z10 * scale, z50 * scale, z90 * scale])
        
        forecast_viz.append({
            "time": target_time,
            "p10": last_price * np.exp(sorted_rets[0]),
            "p50": last_price * np.exp(sorted_rets[1]),
            "p90": last_price * np.exp(sorted_rets[2])
        })

    # Load importance
    imp_df = pd.read_csv("forecasts/latest_ensemble_importance.csv")
    top_features = imp_df["feature"].tolist()

    output = {
        "historical": hist_json,
        "forecast": forecast_viz,
        "drivers": top_features
    }

    with open("dashboard/data.json", "w") as f:
        json.dump(output, f)

    print(f"Generated dashboard/data.json. Drivers: {top_features}")

if __name__ == "__main__":
    main()
