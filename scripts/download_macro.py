import yfinance as yf
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="DX-Y.NYB", help="DXY Index symbol")
    parser.add_argument("--output", default="data/raw/macro_dxy.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Downloading {args.symbol} hourly data...")
    # Hourly data limited to last 730 days by yfinance, but usually enough for recent regimes
    ticker = yf.Ticker(args.symbol)
    df = ticker.history(period="2y", interval="1h")
    
    if df.empty:
        print("Failed to download data.")
        return

    df.reset_index(inplace=True)
    df = df[["Datetime", "Close"]]
    df.columns = ["timestamp", "dxy_close"]
    
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
