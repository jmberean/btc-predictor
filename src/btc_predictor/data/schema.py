from typing import Iterable

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)
    return df


def ensure_timezone(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if tz and tz.upper() != "UTC":
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def require_monotonic(df: pd.DataFrame) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonic increasing")


def in_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df