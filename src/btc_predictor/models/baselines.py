from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List

import numpy as np


@dataclass
class HistoricalQuantileBaseline:
    quantiles: List[float]
    horizons: List
    history_: Dict = None

    def fit(self, y_train: Dict) -> "HistoricalQuantileBaseline":
        self.history_ = {}
        for h in self.horizons:
            values = np.asarray(y_train[h])
            self.history_[h] = {q: np.quantile(values, q) for q in self.quantiles}
        return self

    def predict(self, n: int) -> Dict:
        preds = {}
        for h, qmap in self.history_.items():
            preds[h] = {q: np.repeat(val, n) for q, val in qmap.items()}
        return preds


@dataclass
class RandomWalkBaseline:
    quantiles: List[float]
    horizons: List

    def fit(self, y_train: Dict) -> "RandomWalkBaseline":
        self.sigma_ = {h: float(np.std(y_train[h])) for h in self.horizons}
        self.z_ = {q: NormalDist().inv_cdf(q) for q in self.quantiles}
        return self

    def predict(self, n: int) -> Dict:
        preds = {}
        for h in self.horizons:
            sigma = self.sigma_[h]
            preds[h] = {q: np.repeat(self.z_[q] * sigma, n) for q in self.quantiles}
        return preds
