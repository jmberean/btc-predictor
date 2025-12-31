import os
import subprocess
import sys

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    # Base arguments for both downloads
    base_args = [
        sys.executable, "scripts/download_binance_bulk.py",
        "--asset", "spot",
        "--data-type", "klines",
        "--data-frequency", "1h",
        "--symbols", "BTCUSDT",
        "--output-dir", "data/binance"
    ]

    print("--- Synchronizing Monthly Archives ---")
    run_command(base_args + ["--timeperiod", "monthly"])

    print("\n--- Synchronizing Daily Bars (Current Month) ---")
    run_command(base_args + ["--timeperiod", "daily"])

    print("\n✅ Sync Complete. Data is up to date.")

if __name__ == "__main__":
    main()

