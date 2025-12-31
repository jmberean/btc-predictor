# GEMINI.md

# Working style
- Be thorough and explicit.
- Think step by step; do not skip reasoning steps.
- Prefer depth and correctness over brevity.
- Prefer high-quality, long-term solutions over quick fixes/workarounds.

## Project Overview
`btc-predictor` is a professional-grade, leakage-safe Bitcoin forecasting pipeline. It generates probabilistic 12-hour price trajectories validated against 7 years of market data. The system enforces an "available_at" timestamp contract for all features to ensure zero look-ahead bias.

## Tech Stack
*   **Language:** Python 3.10+
*   **Data Handling:** Pandas, NumPy, CCXT, Binance Bulk Downloader
*   **Machine Learning:** LightGBM (Quantile Objective)
*   **Deep Learning:** PyTorch (LSTM, N-BEATS)
*   **Statistical Models:** Statsmodels (ARIMA), ARCH (GARCH)
*   **Configuration:** PyYAML
*   **Visualization:** TradingView-powered Lightweight Charts

## Architecture (Production Protocol)
1.  **Causal Data Sync:** Synchronizes monthly archives and daily bars via `scripts/sync_data.py`.
2.  **Source-Level Winsorization:** Clips extreme log returns at 4 standard deviations to ensure mathematical stability for statistical solvers.
3.  **High-Intensity Training:** 
    *   7 Semi-Annual Expanding Window Folds (2018-2025).
    *   80-Trial Optuna-style hyperparameter tuning per model.
    *   24h Purge / 168h Embargo to eliminate overlap leakage.
4.  **Ensemble Inference:** Combines LightGBM, ARIMA, and GARCH (with LSTM/N-BEATS support) into a unified 12-hour trajectory.
5.  **Conformal Calibration:** Calibrates P10-P90 bounds based on historical coverage to maintain "honesty."
6.  **Interactive Dashboard:** TradingView UI with dynamic horizon handling and feature driver insights.

## Key Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### Data Synchronization
```bash
python scripts/sync_data.py
```

### High-Intensity Training
```bash
# Execute definitive 7-year validation
python scripts/train.py --config configs/binance_bulk.yaml
```

### Production Inference
```bash
# Generate 12h forecast using best ensemble folds
python scripts/infer.py --config configs/binance_bulk.yaml --model <model_paths> --output forecasts/latest_ensemble.csv
```

### Visualization
```bash
# Prepare dashboard data and launch server
python scripts/prepare_viz.py --artifacts artifacts/<run_id>
python -m http.server 8000
```

## Robustness Features
*   **Mathematical Stability:** ARIMA and GARCH feature high-quality pre-validation (Variance Floor, Unique Value check) and fallback mechanisms (Mean/Constant Variance) to survive extreme market segments.
*   **Clean Logs:** Integrated session logging in `logs/` directory captures all `stdout`, `stderr`, and warnings into unique, timestamped files.
*   **Feature Integrity:** `LightGBMQuantileModel` preserves feature names throughout the lifecycle to ensure prediction consistency and clean logs.

## Project Structure
*   **src/btc_predictor/utils/logging.py**: Core logging infrastructure.
*   **src/btc_predictor/models/**: Robust model implementations with built-in stability.
*   **src/btc_predictor/features/engineering.py**: Source of Truth for outlier mitigation.
*   **dashboard/**: UI implementation utilizing TradingView charts.
*   **logs/**: Persistent history of all pipeline sessions.

## Development Notes
*   **Active Session Focus:** Forecasts are optimized for the 1-12h window where predictive Alpha is maximum.
*   **Multiprocessing:** Training uses `n_jobs=-1` for parallel fold execution. Note for macOS: Ensure `num_workers=0` for Deep Learning loaders to prevent MPS instability.
*   **Leakage Prevention:** Rigorous 24h Purge + 168h Embargo is non-negotiable for all validation runs.