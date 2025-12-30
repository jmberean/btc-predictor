from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    formula: str
    inputs: List[str]
    source: str
    delay: pd.Timedelta


def build_feature_catalog(base_delay: pd.Timedelta) -> List[FeatureSpec]:
    return [
        FeatureSpec("log_return_1", "log(close/close.shift(1))", ["close"], "ohlcv", base_delay),
        FeatureSpec("range_pct", "(high-low)/close", ["high", "low", "close"], "ohlcv", base_delay),
        FeatureSpec("rsi", "RSI(close, window)", ["close"], "ohlcv", base_delay),
        FeatureSpec("atr", "ATR(high, low, close)", ["high", "low", "close"], "ohlcv", base_delay),
        FeatureSpec("realized_vol", "sqrt(sum(ret^2)) over window", ["close"], "ohlcv", base_delay),
        FeatureSpec("momentum", "close/close.shift(n)-1", ["close"], "ohlcv", base_delay),
        FeatureSpec("volume_z", "(volume-mean)/std", ["volume"], "ohlcv", base_delay),
        FeatureSpec("drawdown", "close/rolling_max-1", ["close"], "ohlcv", base_delay),
        FeatureSpec("session", "hour-of-day one-hot", ["timestamp"], "calendar", base_delay),
        FeatureSpec("dow", "day-of-week", ["timestamp"], "calendar", base_delay),
    ]


def add_available_at(df: pd.DataFrame, delay: pd.Timedelta) -> pd.DataFrame:
    df = df.copy()
    df["available_at"] = df["timestamp"] + delay
    return df


def combine_available_at(base_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    df = df.merge(external_df, on="timestamp", how="left", suffixes=("", "_ext"))
    if "available_at_ext" in df.columns:
        df["available_at"] = df[["available_at", "available_at_ext"]].max(axis=1)
        df = df.drop(columns=["available_at_ext"])
    return df


def assert_point_in_time(df: pd.DataFrame) -> None:
    if (df["available_at"] < df["timestamp"]).any():
        raise ValueError("available_at cannot be earlier than timestamp")
    if not df["available_at"].is_monotonic_increasing:
        raise ValueError("available_at must be monotonic increasing")