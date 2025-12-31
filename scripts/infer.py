import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from btc_predictor.inference.predict import run_inference
from btc_predictor.utils.logging import setup_logging


def main():
    setup_logging("infer")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, nargs="+")
    parser.add_argument("--asof", default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--output", default="forecast.csv")
    args = parser.parse_args()

    out_path = run_inference(args.config, args.model, args.asof, args.output, weights=args.weights)
    print(out_path)


if __name__ == "__main__":
    main()
