import ccxt
import pandas as pd
from datetime import datetime
import os
import glob

def check_local():
    print("--- Checking Local Files ---")
    local_path = "data/binance/data/spot/monthly/klines/BTCUSDT/1h/*.csv"
    files = sorted(glob.glob(local_path))
    if not files:
        print("No local CSV files found in the monthly directory.")
        return None
    
    latest_file = files[-1]
    print(f"Latest local file: {os.path.basename(latest_file)}")
    
    # Read last line of the latest file
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].split(',')
        # Binance timestamp is typically index 0
        ts = int(last_line[0])
        # Detect if micro or milli
        unit = 'ms' if ts < 1e13 else 'us'
        dt = pd.to_datetime(ts, unit=unit if unit == 'ms' else 'us', utc=True)
        print(f"Latest timestamp in local file: {dt}")
        return dt

def check_remote():
    print("\n--- Checking Binance US API (Live) ---")
    exchange = ccxt.binanceus()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=5)
    latest_api_ts = ohlcv[-1][0]
    latest_api_dt = pd.to_datetime(latest_api_ts, unit='ms', utc=True)
    latest_price = ohlcv[-1][4]
    print(f"Current Time (Local Machine): {datetime.now()}")
    print(f"Latest timestamp on Binance API: {latest_api_dt}")
    print(f"Current BTC Price: ${latest_price}")
    return latest_api_dt

if __name__ == "__main__":
    local_dt = check_local()
    remote_dt = check_remote()
    
    if local_dt and remote_dt:
        diff = remote_dt - local_dt
        print(f"\nGap between local data and live API: {diff}")
        if diff.days > 0:
            print(f"NOTICE: You are missing {diff.days} days of data.")
        else:
            print("Local data is up to date with the API.")
