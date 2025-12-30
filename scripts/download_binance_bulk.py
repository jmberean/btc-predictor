import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from binance_bulk_downloader.downloader import BinanceBulkDownloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="spot", choices=["spot", "um", "cm", "option"])
    parser.add_argument("--data-type", default="klines")
    parser.add_argument("--data-frequency", default="1h")
    parser.add_argument("--timeperiod", default="monthly", choices=["daily", "monthly"])
    parser.add_argument("--symbols", default="BTCUSDT")
    parser.add_argument("--output-dir", default="data/binance")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    symbols = symbols if symbols else None

    downloader = BinanceBulkDownloader(
        destination_dir=args.output_dir,
        data_type=args.data_type,
        data_frequency=args.data_frequency,
        asset=args.asset,
        timeperiod_per_file=args.timeperiod,
        symbols=symbols,
    )
    downloader.run_download()


if __name__ == "__main__":
    main()
