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
        forecasts = self.result_.forecast(horizon=1)
        sigma1 = np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100.0
        preds = {}
        for h in self.horizons:
            step = self.base_steps[h]
            sigma = sigma1 * np.sqrt(step)
            preds[h] = {}
            for q in self.quantiles:
                z = NormalDist().inv_cdf(q)
                preds[h][q] = np.repeat(z * sigma, n)
        return preds