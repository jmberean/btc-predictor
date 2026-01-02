import os
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from btc_predictor.utils.logging import setup_logging

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

import yaml

def main():
    setup_logging("sync_data")
    
    # Load config to get the correct timeframe
    with open("configs/binance_bulk.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    timeframe = cfg["data"]["binance_bulk"].get("timeframe", "1h")

    # Base arguments for both downloads
    base_args = [
        sys.executable, "scripts/download_binance_bulk.py",
        "--asset", "spot",
        "--data-type", "klines",
        "--data-frequency", timeframe,
        "--symbols", "BTCUSDT",
        "--output-dir", "data/binance"
    ]

    print("--- Synchronizing Monthly Archives (Spot) ---")
    run_command(base_args + ["--timeperiod", "monthly"])

    print("\n--- Synchronizing Daily Bars (Spot - Current Month) ---")
    run_command(base_args + ["--timeperiod", "daily"])

    # --- Futures Data (Funding Rate) ---
    funding_args = [
        sys.executable, "scripts/download_binance_bulk.py",
        "--asset", "um",
        "--data-type", "fundingRate",
        "--symbols", "BTCUSDT",
        "--output-dir", "data/binance"
    ]
    print("\n--- Synchronizing Funding Rates (Futures) ---")
    run_command(funding_args + ["--timeperiod", "monthly"])
    run_command(funding_args + ["--timeperiod", "daily"])

    # --- Futures Data (Metrics: Open Interest, L/S Ratio) ---
    metrics_args = [
        sys.executable, "scripts/download_binance_bulk.py",
        "--asset", "um",
        "--data-type", "metrics",
        "--symbols", "BTCUSDT",
        "--output-dir", "data/binance"
    ]
    print("\n--- Synchronizing Metrics (Open Interest) ---")
    run_command(metrics_args + ["--timeperiod", "monthly"])
    run_command(metrics_args + ["--timeperiod", "daily"])

    # --- Futures Data (Mark & Index Price Klines) ---
    for kline_type in ["markPriceKlines", "indexPriceKlines"]:
        print(f"\n--- Synchronizing {kline_type} (Futures) ---")
        futures_kline_args = [
            sys.executable, "scripts/download_binance_bulk.py",
            "--asset", "um",
            "--data-type", kline_type,
            "--data-frequency", timeframe,
            "--symbols", "BTCUSDT",
            "--output-dir", "data/binance"
        ]
        run_command(futures_kline_args + ["--timeperiod", "monthly"])

    print("\n✅ Sync Complete. Data is up to date.")

if __name__ == "__main__":
    main()

