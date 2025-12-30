from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List

import numpy as np
from arch import arch_model


@dataclass
class GARCHQuantileModel:
    quantiles: List[float]
    horizons: List
    base_steps: Dict

    def __post_init__(self):
        self.result_ = None

    def fit(self, y_train: Dict) -> "GARCHQuantileModel":
        min_horizon = min(self.horizons, key=lambda h: self.base_steps[h])
        series = np.asarray(y_train[min_horizon])
        model = arch_model(series * 100, vol="Garch", p=1, q=1, dist="t")
        self.result_ = model.fit(disp="off")
        return self

    def predict(self, n: int) -> Dict:
        max_steps = max(self.base_steps.values())
        total_steps = n + max_steps - 1
        forecasts = self.result_.forecast(horizon=total_steps)
        var_path = np.asarray(forecasts.variance.iloc[-1].values, dtype=float)
        sigma_path = np.sqrt(var_path) / 100.0
        preds = {}
        for h in self.horizons:
            step = self.base_steps[h]
            start = step - 1
            sigma = sigma_path[start : start + n]
            preds[h] = {}
            for q in self.quantiles:
                z = NormalDist().inv_cdf(q)
                preds[h][q] = z * sigma
        return preds
