# GEMINI.md

# Working style
- Be thorough and explicit.
- Think step by step; do not skip reasoning steps.
- Prefer depth and correctness over brevity.
- Prefer high-quality, long-term solutions over quick fixes/workarounds.

## Project Overview
`btc-predictor` is a leakage-safe, walk-forward Bitcoin forecasting pipeline. It focuses on generating probabilistic forecasts (quantiles) for BTC price movements. The system is designed to be rigorous about data availability to prevent look-ahead bias (leakage), utilizing an "available_at" timestamp contract for all features.

## Tech Stack
*   **Language:** Python 3.10+
*   **Data Handling:** Pandas, NumPy, CCXT (for live data), Binance Bulk Downloader
*   **Machine Learning:** Scikit-learn, LightGBM
*   **Deep Learning:** PyTorch (LSTM, N-BEATS implementations)
*   **Statistical Models:** Statsmodels, ARCH (GARCH)
*   **Configuration:** PyYAML

## Architecture
The pipeline follows a strict sequence to ensure data integrity:
1.  **Data Ingestion:** Raw OHLCV data is loaded from CSVs or downloaded via `scripts/download_binance_bulk.py`.
2.  **Schema Validation:** Enforces `timestamp`, `open`, `high`, `low`, `close`, `volume` columns.
3.  **Feature Engineering:** Computes features (returns, volatility, momentum) and assigns an `available_at` timestamp (default: 5 min delay).
4.  **Dataset Construction:** Builds supervised learning pairs (features + targets) respecting horizon gaps.
5.  **Walk-Forward Validation:** Uses expanding or rolling windows with an "embargo" period >= max forecast horizon to prevent leakage.
6.  **Model Training:** Trains baselines, classical statistical models, tree-based models (LightGBM), and deep learning models (LSTM, N-BEATS).
7.  **Evaluation:** Computes metrics (MAE, RMSE, Pinball Loss) and generates reliability curves.
8.  **Artifacts:** Saves models, metrics, and plots to `artifacts/<run_id>/`.

## Key Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Data Download (Binance Bulk)
```bash
export PYTHONPATH=src
python scripts/download_binance_bulk.py --asset spot --data-type klines --data-frequency 1h --symbols BTCUSDT --timeperiod monthly --output-dir data/binance
```

### Training
```bash
export PYTHONPATH=src
# Small config for testing/dev
python scripts/train.py --config configs/small.yaml
# Full run with Binance data
python scripts/train.py --config configs/binance_bulk.yaml
```

### Evaluation
```bash
export PYTHONPATH=src
python scripts/evaluate.py --artifacts artifacts/<run_id>
```

### Inference
```bash
export PYTHONPATH=src
python scripts/infer.py --config configs/small.yaml --model artifacts/<run_id>/models/lightgbm_fold0.joblib
```

## Project Structure
*   `configs/`: YAML configuration files defining model hyperparameters, data paths, and evaluation settings.
*   `scripts/`: CLI entry points for pipeline stages (`train.py`, `evaluate.py`, `infer.py`, `download_binance_bulk.py`).
*   `src/btc_predictor/`:
    *   `config.py`: Configuration loading and validation.
    *   `data/`: Data ingestion, schema definition, and external data handling.
    *   `features/`: Feature engineering logic and `available_at` management.
    *   `models/`: Model implementations (`deep.py` for LSTM, `sota.py` for N-BEATS, `tree.py` for LightGBM, `baselines.py`, `arima.py`, `garch.py`).
    *   `training/`: Main training loop and cross-validation logic.
    *   `evaluation/`: Metric calculations and plotting.
    *   `inference/`: Real-time prediction logic.

## Development Notes
*   **Leakage Prevention:** Critical. All features must have an associated `available_at` timestamp. Training splits respect this strictly.
*   **Deep Learning Backend:** The code automatically detects `cuda`, `mps` (macOS), or `cpu`.
    *   *Note for macOS users:* If using MPS, ensure `num_workers=0` in data loaders to avoid PyTorch multiprocessing crashes.
*   **Quantile Regression:** Models are trained to predict specific quantiles (e.g., 0.1, 0.5, 0.9) rather than just the mean, enabling probabilistic confidence intervals.
*   **Code Style:** Follows standard Python conventions. Type hinting is encouraged.

## Model Roster
*   **Baselines:** Naive quantiles, Random Walk.
*   **Classical:** ARIMA, GARCH.
*   **Tree:** LightGBM (Quantile Objective).
*   **Deep Learning:**
    *   **LSTM:** Standard Long Short-Term Memory network for quantile regression.
    *   **N-BEATS:** Neural Basis Expansion Analysis for Time Series (implemented in `src/btc_predictor/models/sota.py`).
