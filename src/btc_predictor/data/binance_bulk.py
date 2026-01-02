import glob
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

from btc_predictor.config import parse_timedelta
from btc_predictor.data.schema import ensure_timezone, validate_ohlcv

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

DTYPE_MAP = {
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "float32",
    "quote_asset_volume": "float32",
    "number_of_trades": "int32",
    "taker_buy_base_asset_volume": "float32",
    "taker_buy_quote_asset_volume": "float32",
}


def _resolve_files(path_or_glob: str | List[str]) -> List[str]:
    if isinstance(path_or_glob, list):
        all_files = []
        for p in path_or_glob:
            all_files.extend(_resolve_files(p))
        return sorted(list(set(all_files)))
    
    path = Path(path_or_glob)
    if path.is_dir():
        files = list(path.rglob("*.zip")) + list(path.rglob("*.csv"))
        return sorted(str(p) for p in files)
    return sorted(glob.glob(path_or_glob))


def _read_csv(path: str, columns: List[str], usecols: Optional[List[int]] = None) -> pd.DataFrame:
    # First, peek at the first row to detect headers
    try:
        peek = pd.read_csv(path, header=None, nrows=1)
        if peek.empty:
            return pd.DataFrame(columns=columns)
        
        first_val = str(peek.iloc[0, 0]).lower()
        has_header = "open_time" in first_val or "funding" in first_val or "create_time" in first_val
        skiprows = 1 if has_header else 0
        
        df = pd.read_csv(
            path,
            header=None,
            names=columns,
            usecols=usecols,
            dtype=DTYPE_MAP,
            engine="c",
            skiprows=skiprows
        )
        return df
    except (ValueError, TypeError):
        # Fallback if dtype casting fails on mixed rows (rare with skiprows)
        df = pd.read_csv(path, header=None, names=columns, usecols=usecols)
        if df.empty:
            return df
        first = str(df.iloc[0, 0]).lower()
        if "open_time" in first or "funding" in first or "create_time" in first:
            df = df.iloc[1:].reset_index(drop=True)
        # Final attempt to cast to optimized types
        for col, dtype in DTYPE_MAP.items():
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        return df


def _read_zip(path: str, columns: List[str], usecols: Optional[List[int]] = None) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                frames.append(pd.read_csv(f, header=None, names=columns, usecols=usecols))
    if not frames:
        return pd.DataFrame(columns=columns)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    first = str(df.iloc[0, 0]).lower()
    if "open_time" in first or "funding" in first or "create_time" in first:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _read_files(path_or_glob: str, columns: List[str], usecols: Optional[List[int]] = None) -> pd.DataFrame:
    files = _resolve_files(path_or_glob)
    frames = []
    for path in files:
        if path.lower().endswith(".zip"):
            frames.append(_read_zip(path, columns, usecols=usecols))
        elif path.lower().endswith(".csv"):
            frames.append(_read_csv(path, columns, usecols=usecols))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _to_datetime(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    max_val = values.dropna().max()
    if pd.isna(max_val):
        return pd.to_datetime(values, unit="ms", utc=True)

    if max_val > 1e14:
        us_mask = values > 1e14
        dt_ms = pd.to_datetime(values.where(~us_mask), unit="ms", utc=True)
        dt_us = pd.to_datetime(values.where(us_mask), unit="us", utc=True)
        return dt_ms.fillna(dt_us)

    if max_val > 1e12:
        unit = "ms"
    elif max_val > 1e9:
        unit = "s"
    else:
        unit = "ms"
    return pd.to_datetime(values, unit=unit, utc=True)


def _filter_timeframe(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df["timestamp"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[df["timestamp"] <= end_ts]
    return df


import hashlib
import os

def _get_cache_path(files: List[str], timeframe: str, tz: str) -> Path:
    cache_dir = Path("data/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a hash based on file paths, timeframe, and timezone
    content_str = "".join(files) + timeframe + tz
    file_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    return cache_dir / f"binance_bulk_{file_hash}.parquet"


def load_binance_bulk_klines(
    path_or_glob: str,
    timeframe: str,
    tz: str = "UTC",
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    files = _resolve_files(path_or_glob)
    if not files:
        raise FileNotFoundError(f"No kline files found for {path_or_glob}")

    cache_path = _get_cache_path(files, timeframe, tz)

    if use_cache and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            # Apply runtime filters even if cached (cache stores full range for these files)
            df = _filter_timeframe(df, start, end)
            return df
        except Exception:
            # If cache is corrupted, fall back to reload
            pass

    # Existing loading logic using specific files list instead of resolving again
    frames = []
    for path in files:
        if path.lower().endswith(".zip"):
            frames.append(_read_zip(path, KLINE_COLUMNS, usecols=[0, 1, 2, 3, 4, 5]))
        elif path.lower().endswith(".csv"):
            frames.append(_read_csv(path, KLINE_COLUMNS, usecols=[0, 1, 2, 3, 4, 5]))
            
    if not frames:
        raw = pd.DataFrame(columns=KLINE_COLUMNS)
    else:
        raw = pd.concat(frames, ignore_index=True)

    if raw.empty:
         raise FileNotFoundError(f"No valid data found in files for {path_or_glob}")

    raw["open_time"] = _to_datetime(raw["open_time"])
    delta = parse_timedelta(timeframe)
    raw["timestamp"] = raw["open_time"] + delta

    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw[["timestamp", "open", "high", "low", "close", "volume"]]
    df = ensure_timezone(df, tz=tz)
    df = validate_ohlcv(df)
    
    if use_cache:
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass

    df = _filter_timeframe(df, start, end)
    return df


def load_binance_bulk_kline_feature(
    path_or_glob: str,
    timeframe: str,
    prefix: str,
    tz: str = "UTC",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    raw = _read_files(path_or_glob, KLINE_COLUMNS, usecols=[0, 4])
    if raw.empty:
        raise FileNotFoundError(f"No kline feature files found for {path_or_glob}")

    raw["open_time"] = _to_datetime(raw["open_time"])
    delta = parse_timedelta(timeframe)
    raw["timestamp"] = raw["open_time"] + delta
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")

    df = raw[["timestamp", "close"]].rename(columns={"close": f"{prefix}_close"})
    df = ensure_timezone(df, tz=tz)
    df = _filter_timeframe(df, start, end)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def load_binance_bulk_funding_rate(
    path_or_glob: str,
    tz: str = "UTC",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    raw = _read_files(path_or_glob, ["col1", "col2", "col3", "col4"])
    if raw.empty:
        raise FileNotFoundError(f"No funding rate files found for {path_or_glob}")

    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    col0 = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    col1 = pd.to_numeric(raw.iloc[:, 1], errors="coerce") if raw.shape[1] > 1 else None
    if col0.max(skipna=True) > 1e12:
        time_col = 0
    elif col1 is not None and col1.max(skipna=True) > 1e12:
        time_col = 1
    else:
        time_col = 0

    rate_col = 2 if raw.shape[1] > 2 else 1
    raw = raw.rename(
        columns={
            raw.columns[time_col]: "funding_time",
            raw.columns[rate_col]: "funding_rate",
        }
    )

    raw["funding_time"] = _to_datetime(raw["funding_time"])
    raw["funding_rate"] = pd.to_numeric(raw["funding_rate"], errors="coerce")

    df = raw[["funding_time", "funding_rate"]].rename(columns={"funding_time": "timestamp"})
    df = ensure_timezone(df, tz=tz)
    df = _filter_timeframe(df, start, end)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def load_binance_bulk_metrics(
    path_or_glob: str,
    tz: str = "UTC",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    # Schema: create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, 
    #         sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio
    cols = [
        "create_time", 
        "symbol", 
        "sum_open_interest", 
        "sum_open_interest_value", 
        "count_toptrader_long_short_ratio", 
        "sum_toptrader_long_short_ratio", 
        "count_long_short_ratio", 
        "sum_taker_long_short_vol_ratio"
    ]
    
    # We read all columns initially to handle parsing
    files = _resolve_files(path_or_glob)
    print(f"DEBUG: Found {len(files)} metrics files in {path_or_glob}")
    
    raw = _read_files(path_or_glob, cols)
    print(f"DEBUG: Raw metrics shape: {raw.shape}")
    if not raw.empty:
        print(f"DEBUG: Raw head:\n{raw.head()}")
    
    if raw.empty:
        raise FileNotFoundError(f"No metrics files found for {path_or_glob}")

    # Ensure create_time is valid (ISO format in metrics files)
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
    
    # Convert numeric columns
    numeric_cols = [
        "sum_open_interest", 
        "sum_open_interest_value", 
        "count_long_short_ratio", 
        "sum_taker_long_short_vol_ratio"
    ]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Rename and select
    df = raw.rename(columns={
        "create_time": "timestamp",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_long_short_ratio": "ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_ls_ratio"
    })
    
    df = df[["timestamp", "open_interest", "open_interest_value", "ls_ratio", "taker_ls_ratio"]]
    df = ensure_timezone(df, tz=tz)
    df = _filter_timeframe(df, start, end)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df
