from dataclasses import dataclass
from typing import Dict, List

import lightgbm as lgb
import numpy as np


@dataclass
class LightGBMQuantileModel:
    params: Dict
    quantiles: List[float]
    horizons: List

    def __post_init__(self):
        self.models_ = {}

    def fit(self, x, y: Dict) -> "LightGBMQuantileModel":
        self.models_ = {}
        for h in self.horizons:
            self.models_[h] = {}
            for q in self.quantiles:
                model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
                model.fit(x, y[h])
                self.models_[h][q] = model
        return self

    def predict(self, x) -> Dict:
        preds = {}
        for h in self.horizons:
            preds[h] = {q: self.models_[h][q].predict(x) for q in self.quantiles}
        return preds