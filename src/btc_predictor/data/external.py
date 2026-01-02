from typing import Dict, List

import pandas as pd

from btc_predictor.data.binance_bulk import (
    load_binance_bulk_funding_rate,
    load_binance_bulk_kline_feature,
    load_binance_bulk_metrics,
)


def load_external_features(cfg: Dict) -> List[pd.DataFrame]:
    ext_cfg = cfg.get("external_data", {})
    if not ext_cfg:
        return []

    tz = cfg.get("data", {}).get("timezone", "UTC")
    start = cfg.get("data", {}).get("start")
    end = cfg.get("data", {}).get("end")
    dfs: List[pd.DataFrame] = []

    funding_cfg = ext_cfg.get("funding_rate")
    if funding_cfg and funding_cfg.get("path"):
        df = load_binance_bulk_funding_rate(
            funding_cfg["path"],
            tz=tz,
            start=funding_cfg.get("start", start),
            end=funding_cfg.get("end", end),
        )
        if df.empty or not df["funding_rate"].notna().any():
            df = None
        delay = pd.Timedelta(minutes=funding_cfg.get("delay_minutes", 10))
        if df is not None:
            df["available_at"] = df["timestamp"] + delay
            dfs.append(df)

    # NEW: Metrics (Open Interest)
    metrics_cfg = ext_cfg.get("metrics")
    if metrics_cfg and metrics_cfg.get("path"):
        df = load_binance_bulk_metrics(
            metrics_cfg["path"],
            tz=tz,
            start=metrics_cfg.get("start", start),
            end=metrics_cfg.get("end", end),
        )
        if df.empty:
            df = None
        # Metrics are often delayed
        delay = pd.Timedelta(minutes=metrics_cfg.get("delay_minutes", 15))
        if df is not None:
            df["available_at"] = df["timestamp"] + delay
            dfs.append(df)

    mark_cfg = ext_cfg.get("mark_price_klines")

    if mark_cfg and mark_cfg.get("path"):
        df = load_binance_bulk_kline_feature(
            mark_cfg["path"],
            timeframe=mark_cfg.get("timeframe", cfg["features"]["base_frequency"]),
            prefix="mark",
            tz=tz,
            start=mark_cfg.get("start", start),
            end=mark_cfg.get("end", end),
        )
        if df.empty:
            df = None
        delay = pd.Timedelta(minutes=mark_cfg.get("delay_minutes", 2))
        if df is not None:
            df["available_at"] = df["timestamp"] + delay
            dfs.append(df)

    index_cfg = ext_cfg.get("index_price_klines")
    if index_cfg and index_cfg.get("path"):
        df = load_binance_bulk_kline_feature(
            index_cfg["path"],
            timeframe=index_cfg.get("timeframe", cfg["features"]["base_frequency"]),
            prefix="index",
            tz=tz,
            start=index_cfg.get("start", start),
            end=index_cfg.get("end", end),
        )
        if df.empty:
            df = None
        delay = pd.Timedelta(minutes=index_cfg.get("delay_minutes", 2))
        if df is not None:
            df["available_at"] = df["timestamp"] + delay
            dfs.append(df)

    macro_cfg = ext_cfg.get("macro")
    if macro_cfg and macro_cfg.get("path"):
        # Generic macro CSV loader: timestamp, value
        df = pd.read_csv(macro_cfg["path"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns={df.columns[1]: "macro_val"})
        delay = pd.Timedelta(minutes=macro_cfg.get("delay_minutes", 60))
        df["available_at"] = df["timestamp"] + delay
        dfs.append(df)

    return dfs
