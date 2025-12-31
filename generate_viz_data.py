import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path

def load_all_hist():
    monthly_files = sorted(glob.glob("data/binance/data/spot/monthly/klines/BTCUSDT/1h/*.csv"))
    daily_files = sorted(glob.glob("data/binance/data/spot/daily/klines/BTCUSDT/1h/*.csv"))
    
    all_files = monthly_files + daily_files
    # Only take last 100 bars for the chart context
    frames = []
    for f in all_files:
        # 0: open_time, 1: open, 2: high, 3: low, 4: close
        df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4], names=["time", "open", "high", "low", "close"])
        frames.append(df)
    
    full_df = pd.concat(frames).tail(100)
    # Detect if micro or milli
    sample_ts = full_df["time"].iloc[0]
    unit = 'us' if sample_ts > 1e14 else 'ms'
    
    full_df["time_dt"] = pd.to_datetime(full_df["time"], unit=unit, utc=True)
    return full_df

# Load history
hist_df = load_all_hist()
last_price = hist_df["close"].iloc[-1]
last_time = int(hist_df["time_dt"].iloc[-1].timestamp()) # Unix seconds

hist_json = []
for row in hist_df.itertuples():
    hist_json.append({
        "time": int(row.time_dt.timestamp()),
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close
    })

# Load ensemble forecast
forecast_df = pd.read_csv("ensemble_forecast.csv")

forecast_viz = []
# Start at step 0
forecast_viz.append({
    "time": last_time,
    "p10": last_price,
    "p50": last_price,
    "p90": last_price
})

for horizon in ["1h", "2h", "3h", "4h", "5h"]:
    h_int = int(horizon.replace("h", ""))
    target_time = last_time + (h_int * 3600)
    
    p10_ret = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.1)]["y_pred"].values[0]
    p50_ret = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.5)]["y_pred"].values[0]
    p90_ret = forecast_df[(forecast_df["horizon"] == horizon) & (forecast_df["quantile"] == 0.9)]["y_pred"].values[0]
    
    # Enforce strict quantile order for clean viz
    sorted_rets = sorted([p10_ret, p50_ret, p90_ret])
    
    forecast_viz.append({
        "time": target_time,
        "p10": last_price * np.exp(sorted_rets[0]),
        "p50": last_price * np.exp(sorted_rets[1]),
        "p90": last_price * np.exp(sorted_rets[2])
    })

output = {
    "historical": hist_json,
    "forecast": forecast_viz
}

with open("viz_data.json", "w") as f:
    json.dump(output, f)

print(f"Generated viz_data.json with Ensemble data up to {hist_df['time_dt'].iloc[-1]}")
