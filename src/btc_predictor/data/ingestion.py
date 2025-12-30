from typing import Optional

import pandas as pd

from btc_predictor.config import parse_timedelta
from btc_predictor.data.schema import OHLCV_COLUMNS, ensure_timezone, validate_ohlcv


def load_ohlcv_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a timestamp column")
    df = df[OHLCV_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = ensure_timezone(df, tz=tz)
    df = validate_ohlcv(df)
    return df


def fetch_ohlcv_ccxt(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ccxt is required for exchange ingestion") from exc

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    all_rows = []
    tf_delta = parse_timedelta(timeframe)

    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)
        last_ms = rows[-1][0]
        since_ms = last_ms + int(tf_delta.total_seconds() * 1000)
        if end_ts and pd.Timestamp(last_ms, unit="ms", tz="UTC") >= end_ts:
            break

    df = pd.DataFrame(all_rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True) + tf_delta
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = validate_ohlcv(df)
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    return df