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
        # Use rescale=True to let the optimizer handle tiny log returns or large Z-scores
        model = arch_model(series, vol="Garch", p=1, q=1, dist="t", rescale=True)
        self.result_ = model.fit(disp="off")
        return self

    def predict(self, n: int) -> Dict:
        if hasattr(n, "__len__"):
            n = len(n)
            
        if n == 0:
            return {h: {q: np.array([]) for q in self.quantiles} for h in self.horizons}
            
        max_steps = max(self.base_steps.values())
        total_steps = int(n + max_steps)
        forecasts = self.result_.forecast(horizon=total_steps)
        var_path = np.asarray(forecasts.variance.iloc[-1].values, dtype=float)
        
        # Scaling correction: arch_model.rescale_index tracks what multiplier was used
        scale = getattr(self.result_, 'scale', 1.0)
        sigma_path = np.sqrt(var_path) / scale
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
