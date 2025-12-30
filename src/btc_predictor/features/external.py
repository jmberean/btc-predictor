from typing import List

import pandas as pd


def merge_external_features(base_df: pd.DataFrame, external_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not external_dfs:
        return base_df

    df = base_df.sort_values("timestamp").copy()
    for ext in external_dfs:
        ext_sorted = ext.sort_values("timestamp")
        merged = pd.merge_asof(
            df,
            ext_sorted,
            on="timestamp",
            direction="backward",
            suffixes=("", "_ext"),
        )
        if "available_at_ext" in merged.columns:
            merged["available_at"] = merged[["available_at", "available_at_ext"]].max(axis=1)
            merged = merged.drop(columns=["available_at_ext"])
        df = merged
    return df
