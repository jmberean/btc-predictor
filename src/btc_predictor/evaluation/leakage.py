from typing import Dict, List

import numpy as np
import pandas as pd

from btc_predictor.evaluation.metrics import mae
from btc_predictor.models.tree import LightGBMQuantileModel


def time_travel_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict,
) -> Dict[str, float]:
    x_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    x_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Normal model
    model = LightGBMQuantileModel(params=params, quantiles=[0.5], horizons=[target_col])
    model.fit(x_train, {target_col: y_train})
    pred = model.predict(x_test)[target_col][0.5]
    baseline_mae = mae(y_test, pred)

    # Leaky model: shift features forward by 1 step
    x_train_leak = train_df[feature_cols].shift(-1).dropna().values
    y_train_leak = y_train[:-1]
    x_test_leak = test_df[feature_cols].shift(-1).dropna().values
    y_test_leak = y_test[:-1]
    model_leak = LightGBMQuantileModel(params=params, quantiles=[0.5], horizons=[target_col])
    model_leak.fit(x_train_leak, {target_col: y_train_leak})
    pred_leak = model_leak.predict(x_test_leak)[target_col][0.5]
    leak_mae = mae(y_test_leak, pred_leak)

    return {"baseline_mae": baseline_mae, "leaky_mae": leak_mae}