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
        self.model_ = ARIMA(series, order=self.order)
        self.result_ = self.model_.fit()
        return self

    def predict(self, n: int) -> Dict:
        max_steps = max(self.base_steps.values())
        forecast = self.result_.get_forecast(steps=max_steps)
        mean = forecast.predicted_mean
        var = forecast.var_pred_mean
        preds = {}
        for h in self.horizons:
            step = self.base_steps[h]
            mu = mean[step - 1]
            sigma = float(np.sqrt(var[step - 1]))
            preds[h] = {}
            for q in self.quantiles:
                z = NormalDist().inv_cdf(q)
                preds[h][q] = np.repeat(mu + z * sigma, n)
        return preds