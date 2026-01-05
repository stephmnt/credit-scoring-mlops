from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.main import (
    DAYS_EMPLOYED_SENTINEL,
    ENGINEERED_SOURCES,
    IGNORE_FEATURES,
    MISSING_INDICATOR_MIN_RATE,
    OUTLIER_COLUMNS,
    OUTLIER_LOWER_Q,
    OUTLIER_UPPER_Q,
    _apply_correlated_imputation,
    _validate_numeric_inputs,
    add_missingness_indicators,
    apply_outlier_clipping,
    compute_outlier_bounds,
    new_features_creation,
    select_missing_indicator_columns,
)


def preprocess_for_training(df_raw: pd.DataFrame, artifacts: Any) -> pd.DataFrame:
    df = df_raw.copy()

    for col in artifacts.required_input_columns:
        if col not in df.columns:
            df[col] = np.nan

    _validate_numeric_inputs(df, artifacts.numeric_required_columns)

    df['is_train'] = 0
    df['is_test'] = 1
    if 'TARGET' not in df.columns:
        df['TARGET'] = 0

    df = new_features_creation(
        df,
        days_employed_sentinel=DAYS_EMPLOYED_SENTINEL,
        engineered_sources=ENGINEERED_SOURCES,
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.reindex(columns=artifacts.columns_keep, fill_value=np.nan)

    indicator_cols = getattr(artifacts, 'missing_indicator_columns', None) or select_missing_indicator_columns(
        df,
        exclude_cols=set(IGNORE_FEATURES),
        min_missing_rate=MISSING_INDICATOR_MIN_RATE,
    )
    df = add_missingness_indicators(df, indicator_cols)

    outlier_bounds = getattr(artifacts, 'outlier_bounds', {}) or compute_outlier_bounds(
        df,
        OUTLIER_COLUMNS,
        lower_q=OUTLIER_LOWER_Q,
        upper_q=OUTLIER_UPPER_Q,
    )
    df = apply_outlier_clipping(df, outlier_bounds)

    _apply_correlated_imputation(df, artifacts)

    for col, median in artifacts.numeric_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(median)

    for col in artifacts.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    df_hot = pd.get_dummies(df, columns=artifacts.categorical_columns)
    df_hot = df_hot.reindex(columns=artifacts.features_to_scaled, fill_value=0)
    scaled = artifacts.scaler.transform(df_hot)
    return pd.DataFrame(scaled, columns=artifacts.features_to_scaled, index=df.index)
