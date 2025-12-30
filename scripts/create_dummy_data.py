import pandas as pd
import numpy as np

dates = pd.date_range(start="2020-01-01", periods=20000, freq="1h")
df = pd.DataFrame({"timestamp": dates})
df["open"] = 10000 + np.random.randn(len(dates)).cumsum()
df["high"] = df["open"] + np.abs(np.random.randn(len(dates)))
df["low"] = df["open"] - np.abs(np.random.randn(len(dates)))
df["close"] = df["open"] + np.random.randn(len(dates)) * 0.5
df["volume"] = np.random.randint(100, 1000, size=len(dates))

df.to_csv("data/raw/btc_1h.csv", index=False)
print(f"Created dummy data/raw/btc_1h.csv with {len(df)} rows")
