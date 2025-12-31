# BTC-Predictor: Leakage-Safe Ensemble Forecasting

A production-ready Bitcoin forecasting pipeline designed for scientific rigor and trading evaluation. This system uses a weighted ensemble of Gradient Boosting (LightGBM), Recurrent Neural Networks (LSTM), and Signal Decomposition (N-BEATS) to predict continuous 5-hour price trajectories.

## 🚀 End-to-End Workflow

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### 2. Synchronize Data (Live Context)
Get fully up to date by downloading both monthly archives and the latest daily bars in one operation.
```bash
python scripts/sync_data.py
```

### 3. Train Models
Train the full suite of models using the walk-forward validation scheme. This generates the artifacts used for ensembling.
```bash
python scripts/train.py --config configs/binance_bulk.yaml
```
*Artifacts are saved to `artifacts/YYYYMMDD_HHMMSS/`.*

### 4. Backtest & Determine Weights
Evaluate which models are performing best to decide your ensemble weights.
```bash
# Check 1h horizon performance
python scripts/trading_eval.py --artifacts artifacts/<run_id> --horizon 1h
```
*Look for high Sharpe ratios and low Drawdowns.*

### 5. Production Ensemble Inference
Generate a "Consensus Forecast" by combining the models. The example below uses a 50/30/20 weighting based on typical performance.
```bash
python scripts/infer.py \
  --config configs/binance_bulk.yaml \
  --model artifacts/<run_id>/models/nbeats_fold7.joblib \
          artifacts/<run_id>/models/lstm_fold7.joblib \
          artifacts/<run_id>/models/lightgbm_fold7.joblib \
  --weights 0.5 0.3 0.2 \
  --output ensemble_forecast.csv
```

### 6. Interactive Visualization
Sync the latest data with the dashboard and view the results in your browser.
```bash
# Process latest history + ensemble forecast for the UI
python generate_viz_data.py

# Open the TradingView-style dashboard
open continuous_forecast_viz.html
```

## 🛠 Core Components

### Feature Availability Contract
The system uses an `available_at` timestamp for every data point. This ensures that the model only trains on information that was actually published at the time of prediction, strictly preventing "look-ahead" bias (leakage).

### The Ensemble Team
- **N-BEATS**: Decomposes price into Trend and Seasonality. Best for capturing the "wave."
- **LightGBM**: Expert system using technical indicators (RSI, Volatility). Best for identifying "setups."
- **LSTM**: Neural memory that views the market as a continuous flow. Best for "context."

## 📁 Project Structure
- `src/btc_predictor/`: Core logic (data, features, models, training).
- `scripts/`: CLI entry points for the pipeline stages.
- `configs/`: YAML files defining horizons, hyperparameters, and data paths.
- `artifacts/`: Historical run data, saved models, and metrics.
- `continuous_forecast_viz.html`: Professional interactive dashboard.

---
**Disclaimer:** This software is for educational and engineering purposes only. Cryptocurrency trading involves high risk.