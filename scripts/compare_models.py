#!/usr/bin/env python3
"""Compare HistGB, LightGBM, and XGBoost on F1 with consistent preprocessing."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    _validate_numeric_ranges,
    add_missingness_indicators,
    apply_outlier_clipping,
    compute_outlier_bounds,
    load_preprocessor,
    new_features_creation,
    select_missing_indicator_columns,
)


@dataclass
class ModelSpec:
    name: str
    model: Any
    needs_sanitized_features: bool = False


def sanitize_feature_names(columns: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_")
        if not base:
            base = "feature"
        if base[0].isdigit():
            base = f"f_{base}"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cleaned.append(base)
    return cleaned


def best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def preprocess_for_training(df_raw: pd.DataFrame, artifacts: Any) -> pd.DataFrame:
    df = df_raw.copy()

    for col in artifacts.required_input_columns:
        if col not in df.columns:
            df[col] = np.nan

    _validate_numeric_inputs(df, artifacts.numeric_required_columns)
    # Range checks are enforced for inference; skip for training to avoid
    # rejecting legitimate values dropped during preprocessor calibration.

    df["is_train"] = 0
    df["is_test"] = 1
    if "TARGET" not in df.columns:
        df["TARGET"] = 0

    df = new_features_creation(
        df,
        days_employed_sentinel=DAYS_EMPLOYED_SENTINEL,
        engineered_sources=ENGINEERED_SOURCES,
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.reindex(columns=artifacts.columns_keep, fill_value=np.nan)

    indicator_cols = getattr(artifacts, "missing_indicator_columns", None) or select_missing_indicator_columns(
        df,
        exclude_cols=set(IGNORE_FEATURES),
        min_missing_rate=MISSING_INDICATOR_MIN_RATE,
    )
    df = add_missingness_indicators(df, indicator_cols)

    outlier_bounds = getattr(artifacts, "outlier_bounds", {}) or compute_outlier_bounds(
        df,
        OUTLIER_COLUMNS,
        lower_q=OUTLIER_LOWER_Q,
        upper_q=OUTLIER_UPPER_Q,
    )
    df = apply_outlier_clipping(df, outlier_bounds)

    _apply_correlated_imputation(df, artifacts)

    for col, median in artifacts.numeric_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(median)

    for col in artifacts.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df_hot = pd.get_dummies(df, columns=artifacts.categorical_columns)
    df_hot = df_hot.reindex(columns=artifacts.features_to_scaled, fill_value=0)
    scaled = artifacts.scaler.transform(df_hot)
    return pd.DataFrame(scaled, columns=artifacts.features_to_scaled, index=df.index)


def evaluate_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv_splits: int,
    n_jobs: int | None,
) -> dict[str, float]:
    model = spec.model

    if spec.needs_sanitized_features:
        sanitized = sanitize_feature_names(list(X_train.columns))
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train.columns = sanitized
        X_test.columns = sanitized

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]
    best_threshold, best_f1_cv = best_threshold_for_f1(y_train, oof_proba)

    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]

    default_pred = (test_proba >= 0.5).astype(int)
    metrics_default = compute_metrics(y_test, default_pred, test_proba)

    tuned_pred = (test_proba >= best_threshold).astype(int)
    metrics_tuned = compute_metrics(y_test, tuned_pred, test_proba)

    metrics = {
        **{f"default_{k}": v for k, v in metrics_default.items()},
        **{f"tuned_{k}": v for k, v in metrics_tuned.items()},
        "best_threshold": float(best_threshold),
        "best_f1_cv": float(best_f1_cv),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare models on F1 with MLflow logging.")
    parser.add_argument("--data-path", default="data/data_final.parquet")
    parser.add_argument("--artifacts-path", default="artifacts/preprocessor.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--experiment", default="Model-Comparison-F1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.data_path)
    df = df[df["TARGET"].notna()].copy()
    df["TARGET"] = df["TARGET"].astype(int)

    if args.sample_size and args.sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=args.sample_size,
            stratify=df["TARGET"],
            random_state=args.random_state,
        )

    preprocessor = load_preprocessor(Path(args.data_path), Path(args.artifacts_path))

    X_all = preprocess_for_training(df, preprocessor)
    y_all = df["TARGET"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        stratify=y_all,
        random_state=args.random_state,
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    specs = [
        ModelSpec(
            name="HistGB",
            model=HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=200,
                learning_rate=0.05,
                min_samples_leaf=30,
                l2_regularization=0.0,
                class_weight="balanced",
                random_state=args.random_state,
            ),
        ),
        ModelSpec(
            name="LightGBM",
            model=LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=64,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            ),
            needs_sanitized_features=True,
        ),
        ModelSpec(
            name="XGBoost",
            model=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                tree_method="hist",
            ),
            needs_sanitized_features=True,
        ),
    ]

    mlflow.set_experiment(args.experiment)

    for spec in specs:
        with mlflow.start_run(run_name=spec.name):
            mlflow.log_param("model_name", spec.name)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_param("cv_splits", args.cv_splits)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("sample_size", args.sample_size)
            mlflow.log_params(spec.model.get_params())

            metrics = evaluate_model(
                spec,
                X_train,
                y_train,
                X_test,
                y_test,
                cv_splits=args.cv_splits,
                n_jobs=args.n_jobs,
            )
            mlflow.log_metrics(metrics)
            print(f"{spec.name}: tuned_f1={metrics['tuned_f1_score']:.4f} (threshold={metrics['best_threshold']:.3f})")


if __name__ == "__main__":
    main()
