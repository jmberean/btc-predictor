from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from btc_predictor.config import get_base_frequency
from btc_predictor.features.availability import add_available_at
from btc_predictor.features.external import merge_external_features


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_feature_frame(df: pd.DataFrame, cfg: Dict, external_dfs: Optional[List[pd.DataFrame]] = None) -> pd.DataFrame:
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["log_return_1"] = df["log_close"].diff()
    
    # --- High Quality Resolution: Source-Level Outlier Mitigation ---
    # Clip returns at 4 std to prevent numerical instability in ARIMA/GARCH
    # without destroying the signal.
    ret_std = df["log_return_1"].std()
    if not np.isnan(ret_std) and ret_std > 0:
        df["log_return_1"] = df["log_return_1"].clip(lower=-4*ret_std, upper=4*ret_std)

    lags: List[int] = cfg["features"]["lags"]
    rolling_windows: List[int] = cfg["features"]["rolling_windows"]
    rsi_window: int = cfg["features"]["rsi_window"]
    atr_window: int = cfg["features"]["atr_window"]
    regime_window: int = cfg["features"]["regime_window"]

    for lag in lags:
        df[f"log_return_lag_{lag}"] = df["log_return_1"].shift(lag)
        df[f"momentum_{lag}"] = df["close"].pct_change(lag)

    for win in rolling_windows:
        rolled = df["log_return_1"].rolling(win)
        stats = rolled.agg(["mean", "std"])
        df[f"ret_mean_{win}"] = stats["mean"]
        df[f"ret_std_{win}"] = stats["std"]
        
        df[f"realized_vol_{win}"] = np.sqrt((df["log_return_1"] ** 2).rolling(win).sum())
        
        vol_stats = df["volume"].rolling(win).agg(["mean", "std"])
        df[f"volume_z_{win}"] = (
            (df["volume"] - vol_stats["mean"])
            / vol_stats["std"]
        )
        df[f"price_sma_{win}"] = df["close"].rolling(win).mean() / df["close"] - 1

    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["rsi"] = _rsi(df["close"], rsi_window)
    df["atr"] = _atr(df["high"], df["low"], df["close"], atr_window)
    df["drawdown"] = df["close"] / df["close"].rolling(regime_window).max() - 1

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    
    df["volume_momentum_6"] = df["volume"] / (df["volume"].rolling(6).mean() + 1e-8)
    df["volume_momentum_24"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-8)

    vol_ref = df["log_return_1"].rolling(regime_window).std()
    df["vol_regime"] = (vol_ref > vol_ref.rolling(regime_window).median()).astype(int)

    trend_fast = df["close"].rolling(regime_window // 2).mean()
    trend_slow = df["close"].rolling(regime_window).mean()
    df["trend_regime"] = (trend_fast > trend_slow).astype(int)

    delay_minutes = cfg["data"].get("ohlcv_delay_minutes", 5)
    delay = pd.Timedelta(minutes=delay_minutes)
    df = add_available_at(df, delay)

    if external_dfs:
        df = merge_external_features(df, external_dfs)

    # --- Feature Engineering: Derivatives & Metrics ---
    if "open_interest" in df.columns:
        # OI Momentum
        for lag in [1, 4, 24]:
            df[f"oi_chg_{lag}"] = df["open_interest"].pct_change(lag)
        
        # OI Regime (High/Low relative to recent history)
        oi_mean = df["open_interest"].rolling(regime_window).mean()
        oi_std = df["open_interest"].rolling(regime_window).std()
        df["oi_z"] = (df["open_interest"] - oi_mean) / oi_std
        
        # OI / Price Divergence
        # +1: Rising Price + Rising OI (Strong Trend)
        # -1: Rising Price + Falling OI (Weakening/Short Squeeze)
        df["oi_price_corr"] = np.sign(df["open_interest"].diff()) * np.sign(df["close"].diff())

    if "ls_ratio" in df.columns:
        # Crowd Sentiment Z-Score
        ls_mean = df["ls_ratio"].rolling(regime_window).mean()
        ls_std = df["ls_ratio"].rolling(regime_window).std()
        df["ls_ratio_z"] = (df["ls_ratio"] - ls_mean) / ls_std

    if "taker_ls_ratio" in df.columns:
        # Aggressive Flow Z-Score
        taker_mean = df["taker_ls_ratio"].rolling(regime_window).mean()
        taker_std = df["taker_ls_ratio"].rolling(regime_window).std()
        df["taker_buy_sell_z"] = (df["taker_ls_ratio"] - taker_mean) / taker_std

    if "mark_close" in df.columns:
        df["basis_mark"] = df["mark_close"] / df["close"] - 1
    if "index_close" in df.columns:
        df["basis_index"] = df["index_close"] / df["close"] - 1

    df["prediction_time"] = df["available_at"]

    base_frequency = get_base_frequency(cfg)
    df["bar_duration_minutes"] = int(base_frequency.total_seconds() // 60)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    ignore = {
        "timestamp",
        "available_at",
        "prediction_time",
        "max_target_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_close",
        "bar_duration_minutes",
    }
    return [
        c
        for c in df.columns
        if c not in ignore
        and not c.startswith("target_")
        and not c.startswith("target_time_")
        and not c.startswith("y_")
    ]
