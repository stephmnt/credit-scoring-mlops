from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_divide(numer: pd.Series, denom: pd.Series) -> tuple[pd.Series, pd.Series]:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    denom_zero = denom.isna() | (denom == 0)
    return numer / denom.replace(0, np.nan), denom_zero


def new_features_creation(
    df: pd.DataFrame,
    *,
    days_employed_sentinel: int = 365243,
    engineered_sources: Iterable[str] | None = None,
) -> pd.DataFrame:
    df_features = df.copy()
    if engineered_sources is not None:
        for col in engineered_sources:
            if col not in df_features.columns:
                df_features[col] = np.nan

    if "DAYS_EMPLOYED" in df_features.columns:
        if "DAYS_EMPLOYED_ANOM" not in df_features.columns:
            sentinel_mask = df_features["DAYS_EMPLOYED"] == days_employed_sentinel
            df_features["DAYS_EMPLOYED_ANOM"] = sentinel_mask.astype(int)
            df_features.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan

    def _add_ratio(numer_col: str, denom_col: str, ratio_name: str) -> None:
        if numer_col not in df_features.columns or denom_col not in df_features.columns:
            df_features[ratio_name] = np.nan
            df_features[f"DENOM_ZERO_{ratio_name}"] = 1
            return
        ratio, denom_zero = safe_divide(df_features[numer_col], df_features[denom_col])
        df_features[ratio_name] = ratio
        df_features[f"DENOM_ZERO_{ratio_name}"] = denom_zero.astype(int)

    _add_ratio("DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_EMPLOYED_PERC")
    _add_ratio("AMT_INCOME_TOTAL", "AMT_CREDIT", "INCOME_CREDIT_PERC")
    _add_ratio("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_PERSON")
    _add_ratio("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_INCOME_PERC")
    _add_ratio("AMT_ANNUITY", "AMT_CREDIT", "PAYMENT_RATE")

    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_features


def select_missing_indicator_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: set[str] | None = None,
    min_missing_rate: float = 0.0,
) -> list[str]:
    exclude = exclude_cols or set()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    missing_rate = df[numeric_cols].isna().mean()
    cols: list[str] = []
    for col in numeric_cols:
        if col in exclude:
            continue
        if col.startswith(("DENOM_ZERO_", "is_missing_", "is_outlier_")):
            continue
        if col.endswith("_ANOM"):
            continue
        if missing_rate.get(col, 0.0) <= min_missing_rate:
            continue
        cols.append(col)
    return cols


def add_missingness_indicators(
    df: pd.DataFrame, indicator_cols: list[str]
) -> pd.DataFrame:
    for col in indicator_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[f"is_missing_{col}"] = pd.to_numeric(df[col], errors="coerce").isna().astype(int)
    return df


def compute_outlier_bounds(
    df: pd.DataFrame,
    outlier_columns: list[str],
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in outlier_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.dropna().empty:
            continue
        bounds[col] = (
            float(values.quantile(lower_q)),
            float(values.quantile(upper_q)),
        )
    return bounds


def apply_outlier_clipping(
    df: pd.DataFrame, outlier_bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    if not outlier_bounds:
        return df
    df = df.copy()
    for col, (low, high) in outlier_bounds.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values.notna() & ((values < low) | (values > high))
        df[f"is_outlier_{col}"] = mask.astype(int)
        df[col] = values.clip(lower=low, upper=high)
    return df
