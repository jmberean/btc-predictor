# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/btc_predictor/` with subpackages for `data`, `features`, `models`, `training`, `evaluation`, `inference`, and `utils`. CLI entrypoints are in `scripts/` (train, evaluate, infer, leakage check, trading eval). Run configurations live in `configs/*.yaml`. Local datasets go in `data/` (CSV in `data/raw/`, Binance bulk in `data/binance/`), and outputs are written to `artifacts/`. Both `data/` and `artifacts/` are gitignored.

## Build, Test, and Development Commands
Use a venv and install dependencies:
```sh
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```
Common runs (set `PYTHONPATH=src` for imports):
```sh
PYTHONPATH=src python scripts/train.py --config configs/small.yaml
PYTHONPATH=src python scripts/evaluate.py --artifacts artifacts/<run_id>
PYTHONPATH=src python scripts/infer.py --config configs/small.yaml --model artifacts/<run_id>/models/lightgbm_fold0.joblib
PYTHONPATH=src python scripts/leakage_check.py --config configs/small.yaml
PYTHONPATH=src python scripts/trading_eval.py --artifacts artifacts/<run_id> --horizon 1h
```

## Coding Style & Naming Conventions
Python style follows PEP 8 with 4-space indentation. Use `snake_case` for modules, functions, and config keys; `CapWords` for classes; `UPPER_SNAKE_CASE` for constants. New models should live in `src/btc_predictor/models/` and be registered in `src/btc_predictor/models/registry.py`. Add new run profiles as descriptive YAML files under `configs/` (e.g., `medium.yaml`).

## Testing Guidelines
There is no dedicated unit test suite today. Validate changes with the leakage check and evaluation scripts. If you add tests, prefer `pytest` under a `tests/` folder with `test_*.py` naming.

## Commit & Pull Request Guidelines
The history is minimal and has no strict convention; keep commit messages short and imperative. For PRs, include a concise summary, the config file used (e.g., `configs/small.yaml`), key metrics or plots from `artifacts/`, and any new dependencies or data requirements.

## Configuration & Data Tips
Keep `data.start`/`data.end` ranges tight for faster iterations. For Binance bulk downloads, store monthly zips under `data/binance/` and filter date ranges via config rather than editing data files.
