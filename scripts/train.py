import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from btc_predictor.training.train import run_train
from btc_predictor.utils.logging import setup_logging


def main():
    setup_logging("train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out_dir = run_train(args.config)
    print(out_dir)


if __name__ == "__main__":
    main()
