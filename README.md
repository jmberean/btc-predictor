# BTC-Predictor: 10/10 Research Framework

A production-grade Bitcoin forecasting pipeline designed for scientific rigor and actionable trading signals. The system employs a weighted ensemble of Gradient Boosting (LightGBM), Recurrent Neural Networks (LSTM), and Signal Decomposition (N-BEATS) to predict continuous volatility-adjusted price trajectories.

## 🚀 End-to-End Workflow

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### 2. Synchronize Data (Global Context)
Fetch historical monthly archives, current daily bars, and macro-economic indicators (DXY) in one flow.
```bash
python scripts/sync_data.py
python scripts/download_macro.py
```

### 3. Train Models (Purged & Embargoed)
Train the ensemble team using rigorous cross-validation that prevents overlapping correlation leakage.
```bash
python scripts/train.py --config configs/binance_bulk.yaml
```
*Artifacts are saved to `artifacts/YYYYMMDD_HHMMSS/`.*

### 4. Production Ensemble Inference
Generate a "Consensus Forecast" using the 10/10 framework. This produces Z-score normalized outputs and explains the top feature drivers.
```bash
python scripts/infer.py \
  --config configs/binance_bulk.yaml \
  --model artifacts/<run_id>/models/nbeats_fold1.joblib \
          artifacts/<run_id>/models/lstm_fold1.joblib \
          artifacts/<run_id>/models/lightgbm_fold1.joblib \
  --weights 0.4 0.2 0.4 \
  --output forecasts/latest_ensemble.csv
```

### 5. Interactive Pro Dashboard
Prepare the data and launch the TradingView-powered interface.
```bash
# Process raw data for the UI
python scripts/prepare_viz.py

# Start a local server and open the dashboard
# (Run this and visit http://localhost:8000/dashboard)
python -m http.server 8000
```

## 🛠 10/10 Framework Features

- **Causal Integrity:** Uses a 24h Purge and 168h Embargo between training folds to ensure zero look-ahead bias.
- **Volatility Normalization:** Models predict Z-scores rather than raw returns, ensuring stability across different market regimes.
- **Macro Awareness:** Directly incorporates the US Dollar Index (DXY) as a leading indicator.
- **Explainability:** The "Insight Panel" in the UI highlights exactly which features (Momentum, RSI, etc.) are driving the current forecast.
- **Conformal Prediction:** Post-processing layer ensures the P10-P90 "Forecast Cone" is statistically honest.

## 📁 Project Structure
- `dashboard/`: The UI layer (`index.html` + `data.json`).
- `forecasts/`: Raw inference outputs and feature importance data.
- `scripts/`: CLI entry points for syncing, training, and visualizing.
- `src/btc_predictor/`: The core algorithmic engine.
- `artifacts/`: Compressed models and historical backtest metrics.

---
**Disclaimer:** This software is for educational and engineering purposes only. Cryptocurrency trading involves high risk.
