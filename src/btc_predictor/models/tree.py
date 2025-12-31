from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed


def _fit_single_model(
    params: Dict,
    horizon: str,
    quantile: float,
    x: np.ndarray,
    y: np.ndarray,
    eval_set=None,
    eval_metric=None,
    callbacks=None,
    refit_full=False,
    x_full=None,
    y_full=None,
) -> Tuple[str, float, lgb.LGBMRegressor]:
    # Force single threaded for inner model to allow outer parallelism
    local_params = dict(params)
    local_params["n_jobs"] = 1
    # Silence overlap warning: min_data_in_leaf is set, so clear min_child_samples
    if "min_data_in_leaf" in local_params:
        local_params["min_child_samples"] = None
    local_params["verbose"] = -1
    
    model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **local_params)
    model.fit(x, y, eval_set=eval_set, eval_metric=eval_metric, callbacks=callbacks)
    
    if refit_full and x_full is not None and y_full is not None:
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter:
            full_params = dict(local_params)
            full_params["n_estimators"] = int(best_iter)
            model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **full_params)
            model.fit(x_full, y_full)
            
    return horizon, quantile, model


@dataclass
class LightGBMQuantileModel:
    params: Dict
    quantiles: List[float]
    horizons: List

    def __post_init__(self):
        self.models_ = {}
        self.feature_name_ = None

    def fit(
        self,
        x,
        y: Dict,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict] = None,
        early_stopping_rounds: Optional[int] = None,
        refit_full: bool = True,
        x_full: Optional[np.ndarray] = None,
        y_full: Optional[Dict] = None,
    ) -> "LightGBMQuantileModel":
        if hasattr(x, "columns"):
            self.feature_name_ = list(x.columns)
        elif hasattr(x_full, "columns"):
            self.feature_name_ = list(x_full.columns)
            
        self.models_ = {}
        
        tasks = []
        for h in self.horizons:
            for q in self.quantiles:
                eval_set = None
                eval_metric = None
                callbacks = None
                use_early_stop = x_val is not None and y_val is not None and early_stopping_rounds
                if use_early_stop:
                    eval_set = [(x_val, y_val[h])]
                    eval_metric = _pinball_eval(q)
                    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
                
                tasks.append(
                    delayed(_fit_single_model)(
                        self.params,
                        h,
                        q,
                        x,
                        y[h],
                        eval_set,
                        eval_metric,
                        callbacks,
                        use_early_stop and refit_full,
                        x_full,
                        y_full[h] if y_full else None
                    )
                )

        # Use n_jobs=-1 to use all cores for the outer loop
        results = Parallel(n_jobs=-1)(tasks)
        
        for h, q, model in results:
            if h not in self.models_:
                self.models_[h] = {}
            self.models_[h][q] = model
            
        return self

    def predict(self, x) -> Dict:
        import pandas as pd
        if self.feature_name_ and not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self.feature_name_)
            
        preds = {}
        for h in self.horizons:
            preds[h] = {q: self.models_[h][q].predict(x) for q in self.quantiles}
        return preds


def _pinball_eval(q: float):
    def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        diff = y_true - y_pred
        loss = np.maximum(q * diff, (q - 1) * diff)
        return "pinball", float(np.mean(loss)), False

    return _eval
