from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ARIMAQuantileModel:
    order: tuple
    quantiles: List[float]
    horizons: List
    base_steps: Dict

    def __post_init__(self):
        self.model_ = None
        self.result_ = None

    def fit(self, y_train: Dict) -> "ARIMAQuantileModel":
        # Use the shortest horizon series for ARIMA (typically 1-step returns)
        min_horizon = min(self.horizons, key=lambda h: self.base_steps[h])
        series = np.asarray(y_train[min_horizon])
        
        # --- High Quality Resolution: Pre-fitting Validation ---
        # 1. Variance Floor: If series is flat, ARIMA optimizer will explode
        if np.std(series) < 1e-8 or len(np.unique(series)) < 10:
            # Fallback to Mean Model directly if data is low quality
            self.model_ = ARIMA(series, order=(0, 0, 0))
            self.result_ = self.model_.fit()
            return self

        try:
            self.model_ = ARIMA(series, order=self.order)
            self.result_ = self.model_.fit()
        except Exception:
            # Secondary fallback for extreme convergence issues
            self.model_ = ARIMA(series, order=(0, 0, 0))
            self.result_ = self.model_.fit()
        return self

    def predict(self, n: int) -> Dict:
        if hasattr(n, "__len__"):
            n = len(n)
            
        if n == 0:
            return {h: {q: np.array([]) for q in self.quantiles} for h in self.horizons}
            
        max_steps = max(self.base_steps.values())
        total_steps = n + max_steps
        forecast = self.result_.get_forecast(steps=total_steps)
        mean = np.asarray(forecast.predicted_mean)
        var = np.asarray(forecast.var_pred_mean)
        preds = {}
        for h in self.horizons:
            step = self.base_steps[h]
            # base_steps[h] is 1-indexed (e.g. 1h is 1, 4h is 4)
            # The 0-th forecast mean is the 1-step ahead prediction
            start = step - 1
            mu = mean[start : start + n]
            sigma = np.sqrt(np.maximum(var[start : start + n], 1e-9))
            preds[h] = {}
            for q in self.quantiles:
                z = NormalDist().inv_cdf(q)
                preds[h][q] = mu + z * sigma
        return preds
