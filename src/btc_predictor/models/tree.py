from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _fit_single_model(
    params: Dict,
    horizon: str,
    quantile: float,
    x: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    eval_set=None,
    eval_metric=None,
    callbacks=None,
    refit_full=False,
    x_full=None,
    y_full=None,
    feature_name: Optional[List[str]] = None,
) -> Tuple[str, float, lgb.LGBMRegressor]:
    # Force single threaded for inner model to allow outer parallelism
    local_params = dict(params)
    local_params["n_jobs"] = 1
    # Silence overlap warning: min_data_in_leaf is set, so clear min_child_samples
    if "min_data_in_leaf" in local_params:
        local_params["min_child_samples"] = None
    local_params["verbose"] = -1
    
    # Ensure x is a DataFrame if feature names are provided, for clearer debugging/importance
    if feature_name is not None and not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=feature_name)
    if feature_name is not None and x_full is not None and not isinstance(x_full, pd.DataFrame):
        x_full = pd.DataFrame(x_full, columns=feature_name)

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
    n_jobs: int = -1
    chained: bool = True  # Enable Regressor Chain by default

    def __post_init__(self):
        self.models_ = {}
        self.feature_name_ = None
        self._sorted_horizons = []

    def _sort_horizons(self):
        # Sort horizons by duration (1h, 2h, ... 12h)
        # Assuming horizons are strings like "1h", "2h", etc.
        def _parse_h(h):
            return pd.Timedelta(h)
        self._sorted_horizons = sorted(self.horizons, key=_parse_h)

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
            
        self._sort_horizons()
        self.models_ = {}
        
        # If chained, we must process horizons sequentially
        # If not chained, we can parallelize everything
        
        # Prepare data containers
        # We might need to augment X with previous horizon target (Teacher Forcing)
        
        # Helper to get augmented data
        def get_augmented_x(base_x, prev_h_target):
            if not self.chained or prev_h_target is None:
                return base_x
            
            # Add previous horizon target as feature
            # If base_x is DataFrame
            if isinstance(base_x, pd.DataFrame):
                new_x = base_x.copy()
                new_x["prev_horizon_val"] = prev_h_target
                return new_x
            
            # If base_x is numpy
            return np.column_stack([base_x, prev_h_target])

        # We need to maintain the feature names if we are chaining
        current_feature_names = list(self.feature_name_) if self.feature_name_ else None

        for i, h in enumerate(self._sorted_horizons):
            # Determine previous horizon for chaining
            prev_h = self._sorted_horizons[i-1] if i > 0 else None
            
            # Prepare Training Data for this horizon
            x_train_h = x
            x_val_h = x_val
            x_full_h = x_full
            
            aug_feature_names = list(current_feature_names) if current_feature_names else None
            
            if self.chained and prev_h:
                # Augment with Teacher Forcing (True Previous Target)
                x_train_h = get_augmented_x(x, y[prev_h])
                
                if x_val is not None:
                    x_val_h = get_augmented_x(x_val, y_val[prev_h])
                    
                if x_full is not None:
                    x_full_h = get_augmented_x(x_full, y_full[prev_h])

                if aug_feature_names:
                    aug_feature_names.append("prev_horizon_val")

            # Parallelize Quantiles for this Horizon
            tasks = []
            for q in self.quantiles:
                eval_set = None
                eval_metric = None
                callbacks = None
                use_early_stop = x_val_h is not None and y_val is not None and early_stopping_rounds
                if use_early_stop:
                    eval_set = [(x_val_h, y_val[h])]
                    eval_metric = _pinball_eval(q)
                    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
                
                tasks.append(
                    delayed(_fit_single_model)(
                        self.params,
                        h,
                        q,
                        x_train_h,
                        y[h],
                        eval_set,
                        eval_metric,
                        callbacks,
                        use_early_stop and refit_full,
                        x_full_h,
                        y_full[h] if y_full else None,
                        aug_feature_names
                    )
                )

            # Execute tasks for this horizon
            # If chained, we must wait for this horizon to finish before considering next?
            # Actually, with Teacher Forcing, we use GROUND TRUTH y[prev_h], 
            # so we DON'T need the fitted model of prev_h to train current_h.
            # So we CAN parallelize everything! 
            # UNLESS we used predicted values (Recursive). 
            # But we chose Teacher Forcing. 
            # So we can collect all tasks and run them at once?
            # YES.
            
            # However, let's keep the loop structure clear, but collect tasks.
            pass # tasks are collected below
        
        # COLLECT ALL TASKS
        all_tasks = []
        
        # Re-loop to build tasks properly
        for i, h in enumerate(self._sorted_horizons):
            prev_h = self._sorted_horizons[i-1] if i > 0 else None
            
            # Re-construct data (this is cheap, just references)
            x_train_h = x
            x_val_h = x_val
            x_full_h = x_full
            
            aug_feature_names = list(current_feature_names) if current_feature_names else None
            
            if self.chained and prev_h:
                x_train_h = get_augmented_x(x, y[prev_h])
                if x_val is not None:
                    x_val_h = get_augmented_x(x_val, y_val[prev_h])
                if x_full is not None:
                    x_full_h = get_augmented_x(x_full, y_full[prev_h])
                if aug_feature_names:
                    aug_feature_names.append("prev_horizon_val")
            
            for q in self.quantiles:
                eval_set = None
                eval_metric = None
                callbacks = None
                use_early_stop = x_val_h is not None and y_val is not None and early_stopping_rounds
                if use_early_stop:
                    eval_set = [(x_val_h, y_val[h])]
                    eval_metric = _pinball_eval(q)
                    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
                
                all_tasks.append(
                    delayed(_fit_single_model)(
                        self.params,
                        h,
                        q,
                        x_train_h,
                        y[h],
                        eval_set,
                        eval_metric,
                        callbacks,
                        use_early_stop and refit_full,
                        x_full_h,
                        y_full[h] if y_full else None,
                        aug_feature_names
                    )
                )

        results = Parallel(n_jobs=self.n_jobs)(all_tasks)
        
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
        
        # Prediction must be sequential if chained
        if not self._sorted_horizons:
            self._sort_horizons()
            
        # We need a mutable copy to append features if chaining
        # Use a copy to avoid modifying original X
        x_curr = x.copy() if isinstance(x, pd.DataFrame) else x.copy()
        
        for i, h in enumerate(self._sorted_horizons):
            prev_h = self._sorted_horizons[i-1] if i > 0 else None
            
            if self.chained and prev_h:
                # Retrieve prediction from previous horizon (Median/0.5)
                # We need to have computed it already.
                prev_preds = preds[prev_h][0.5]
                
                # Append to x_curr
                if isinstance(x_curr, pd.DataFrame):
                    # Ensure we don't duplicate columns
                    if "prev_horizon_val" in x_curr.columns:
                        x_curr["prev_horizon_val"] = prev_preds
                    else:
                        x_curr["prev_horizon_val"] = prev_preds
                else:
                    # Numpy
                    # We need to replace the last column or append?
                    # In the first step (i=0), x_curr is original X.
                    # In step i=1 (h=2h), we appended 1 col.
                    # In step i=2 (h=3h), we update that col.
                    # So:
                    if i == 1: # Second horizon, first time adding column
                        x_curr = np.column_stack([x_curr, prev_preds])
                    else: # Subsequent horizons, update last column
                        x_curr[:, -1] = prev_preds

            # Predict current horizon
            preds[h] = {}
            for q in self.quantiles:
                preds[h][q] = self.models_[h][q].predict(x_curr)
                
        return preds


@dataclass
class LightGBMClassifierModel:
    params: Dict
    horizons: List
    n_jobs: int = -1

    def __post_init__(self):
        self.models_ = {}

    def fit(self, x, y: Dict, x_val=None, y_val=None, **kwargs):
        self.models_ = {}
        for h in self.horizons:
            # Map -1, 0, 1 -> 0, 1, 2
            y_h = y[h].astype(int) + 1
            
            clf = lgb.LGBMClassifier(**self.params, n_jobs=1)
            clf.fit(x, y_h)
            self.models_[h] = clf
        return self

    def predict(self, x) -> Dict:
        preds = {}
        for h in self.horizons:
            probs = self.models_[h].predict_proba(x)
            # Find probability of Class 2 (which corresponds to Label 1 = Profit)
            # Classes are likely [0, 1, 2] corresponding to [-1, 0, 1]
            classes = self.models_[h].classes_
            
            # Target is index where class == 2
            target_idx = np.where(classes == 2)[0]
            
            if len(target_idx) > 0:
                p_win = probs[:, target_idx[0]]
            else:
                p_win = np.zeros(len(x))
                
            # Store in q=0.5 slot for compatibility
            preds[h] = {0.5: p_win}
        return preds


def _pinball_eval(q: float):
    def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        diff = y_true - y_pred
        loss = np.maximum(q * diff, (q - 1) * diff)
        return "pinball", float(np.mean(loss)), False

    return _eval
