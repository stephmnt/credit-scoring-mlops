from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "data/HistGB_final_model.pkl"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/data_final.parquet"))
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts/preprocessor.joblib"))
DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
CACHE_PREPROCESSOR = os.getenv("CACHE_PREPROCESSOR", "1") != "0"
USE_REDUCED_INPUTS = os.getenv("USE_REDUCED_INPUTS", "1") != "0"
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.85"))
CORRELATION_SAMPLE_SIZE = int(os.getenv("CORRELATION_SAMPLE_SIZE", "50000"))

IGNORE_FEATURES = ["is_train", "is_test", "TARGET", "SK_ID_CURR"]
ENGINEERED_FEATURES = [
    "DAYS_EMPLOYED_PERC",
    "INCOME_CREDIT_PERC",
    "INCOME_PER_PERSON",
    "ANNUITY_INCOME_PERC",
    "PAYMENT_RATE",
]
ENGINEERED_SOURCES = [
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "CNT_FAM_MEMBERS",
    "AMT_ANNUITY",
]
# Top inputs derived from SHAP importance (modeling notebook), limited to application features.
REDUCED_INPUT_FEATURES = [
    "SK_ID_CURR",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "CODE_GENDER",
    "DAYS_EMPLOYED",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "FLAG_OWN_CAR",
]
OUTLIER_COLUMNS = [
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "REGION_POPULATION_RELATIVE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AMT_REQ_CREDIT_BUREAU_QRT",
]


class PredictionRequest(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]]


@dataclass
class PreprocessorArtifacts:
    columns_keep: list[str]
    columns_must_not_missing: list[str]
    numeric_medians: dict[str, float]
    categorical_columns: list[str]
    outlier_maxes: dict[str, float]
    numeric_ranges: dict[str, tuple[float, float]]
    features_to_scaled: list[str]
    scaler: MinMaxScaler
    raw_feature_columns: list[str]
    input_feature_columns: list[str]
    required_raw_columns: list[str]
    required_input_columns: list[str]
    numeric_required_columns: list[str]
    correlated_imputation: dict[str, dict[str, float | str]]


app = FastAPI(title="Credit Scoring API", version="0.1.0")


def new_features_creation(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()
    for col in ENGINEERED_SOURCES:
        if col not in df_features.columns:
            df_features[col] = np.nan
    df_features["DAYS_EMPLOYED_PERC"] = df_features["DAYS_EMPLOYED"] / df_features["DAYS_BIRTH"]
    df_features["INCOME_CREDIT_PERC"] = df_features["AMT_INCOME_TOTAL"] / df_features["AMT_CREDIT"]
    df_features["INCOME_PER_PERSON"] = df_features["AMT_INCOME_TOTAL"] / df_features["CNT_FAM_MEMBERS"]
    df_features["ANNUITY_INCOME_PERC"] = df_features["AMT_ANNUITY"] / df_features["AMT_INCOME_TOTAL"]
    df_features["PAYMENT_RATE"] = df_features["AMT_ANNUITY"] / df_features["AMT_CREDIT"]
    return df_features


def build_preprocessor(data_path: Path) -> PreprocessorArtifacts:
    df = pd.read_parquet(data_path)
    raw_feature_columns = df.columns.tolist()
    input_feature_columns = [c for c in raw_feature_columns if c not in ["is_train", "is_test", "TARGET"]]

    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_rate = df.isna().mean()
    columns_keep = missing_rate[missing_rate < 0.60].index.tolist()
    columns_must_not_missing = missing_rate[missing_rate < 0.010].index.tolist()

    df = df[columns_keep]
    df = df.dropna(subset=columns_must_not_missing)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_medians = df[numeric_cols].median().to_dict()
    df[numeric_cols] = df[numeric_cols].fillna(numeric_medians)

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    df[categorical_columns] = df[categorical_columns].fillna("Unknown")

    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != "XNA"]

    outlier_maxes = {col: df[col].max() for col in OUTLIER_COLUMNS if col in df.columns}
    for col, max_val in outlier_maxes.items():
        df = df[df[col] != max_val]

    numeric_ranges = {}
    for col in numeric_cols:
        if col in df.columns:
            numeric_ranges[col] = (float(df[col].min()), float(df[col].max()))

    df_hot = pd.get_dummies(df, columns=categorical_columns)
    features_to_scaled = [col for col in df_hot.columns if col not in IGNORE_FEATURES]

    scaler = MinMaxScaler()
    scaler.fit(df_hot[features_to_scaled])

    required_raw = set(ENGINEERED_SOURCES)
    required_raw.update(col for col in columns_must_not_missing if col in input_feature_columns)
    required_raw.add("SK_ID_CURR")
    if USE_REDUCED_INPUTS:
        required_input = sorted({col for col in REDUCED_INPUT_FEATURES if col in input_feature_columns})
    else:
        required_input = sorted(required_raw)
    numeric_required = sorted(col for col in required_input if col in numeric_medians)
    correlated_imputation = _build_correlated_imputation(
        df,
        input_feature_columns=input_feature_columns,
        numeric_required=numeric_required,
        threshold=CORRELATION_THRESHOLD,
    )

    return PreprocessorArtifacts(
        columns_keep=columns_keep,
        columns_must_not_missing=columns_must_not_missing,
        numeric_medians={k: float(v) for k, v in numeric_medians.items()},
        categorical_columns=categorical_columns,
        outlier_maxes={k: float(v) for k, v in outlier_maxes.items()},
        numeric_ranges=numeric_ranges,
        features_to_scaled=features_to_scaled,
        scaler=scaler,
        raw_feature_columns=raw_feature_columns,
        input_feature_columns=input_feature_columns,
        required_raw_columns=sorted(required_raw),
        required_input_columns=required_input,
        numeric_required_columns=numeric_required,
        correlated_imputation=correlated_imputation,
    )


def load_preprocessor(data_path: Path, artifacts_path: Path) -> PreprocessorArtifacts:
    if artifacts_path.exists():
        preprocessor = joblib.load(artifacts_path)
        updated = False
        required_updated = False
        if not hasattr(preprocessor, "required_input_columns"):
            if USE_REDUCED_INPUTS:
                required_input = _reduce_input_columns(preprocessor)
            else:
                required_input = preprocessor.required_raw_columns
            preprocessor.required_input_columns = required_input
            required_updated = True
            updated = True
        if not hasattr(preprocessor, "numeric_required_columns"):
            preprocessor.numeric_required_columns = sorted(
                col for col in preprocessor.required_input_columns if col in preprocessor.numeric_medians
            )
            updated = True
        if not hasattr(preprocessor, "numeric_ranges"):
            numeric_ranges = _infer_numeric_ranges_from_scaler(preprocessor)
            if numeric_ranges:
                preprocessor.numeric_ranges = numeric_ranges
                updated = True
            else:
                if not data_path.exists():
                    raise RuntimeError(f"Data file not found to rebuild preprocessor: {data_path}")
                preprocessor = build_preprocessor(data_path)
                updated = True
        if USE_REDUCED_INPUTS:
            reduced = _reduce_input_columns(preprocessor)
            if preprocessor.required_input_columns != reduced:
                preprocessor.required_input_columns = reduced
                required_updated = True
                updated = True
        else:
            if preprocessor.required_input_columns != preprocessor.required_raw_columns:
                preprocessor.required_input_columns = preprocessor.required_raw_columns
                required_updated = True
                updated = True
        desired_numeric_required = sorted(
            col for col in preprocessor.required_input_columns if col in preprocessor.numeric_medians
        )
        if getattr(preprocessor, "numeric_required_columns", None) != desired_numeric_required:
            preprocessor.numeric_required_columns = desired_numeric_required
            updated = True
        if not hasattr(preprocessor, "correlated_imputation") or required_updated:
            if data_path.exists():
                preprocessor.correlated_imputation = _compute_correlated_imputation(data_path, preprocessor)
            else:
                preprocessor.correlated_imputation = {}
            updated = True
        if updated and CACHE_PREPROCESSOR:
            artifacts_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, artifacts_path)
        return preprocessor

    if not data_path.exists():
        raise RuntimeError(f"Data file not found to build preprocessor: {data_path}")

    preprocessor = build_preprocessor(data_path)
    if CACHE_PREPROCESSOR:
        artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, artifacts_path)
    return preprocessor


def load_model(model_path: Path):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _infer_numeric_ranges_from_scaler(preprocessor: PreprocessorArtifacts) -> dict[str, tuple[float, float]]:
    ranges = {}
    scaler = getattr(preprocessor, "scaler", None)
    if scaler is None or not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        return ranges
    for idx, col in enumerate(preprocessor.features_to_scaled):
        if col in preprocessor.numeric_medians:
            ranges[col] = (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
    return ranges


def _build_correlated_imputation(
    df: pd.DataFrame,
    *,
    input_feature_columns: list[str],
    numeric_required: list[str],
    threshold: float,
) -> dict[str, dict[str, float | str]]:
    if not numeric_required:
        return {}
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col in input_feature_columns and col not in {"TARGET", "is_train", "is_test", "SK_ID_CURR"}
    ]
    if not numeric_cols:
        return {}
    df_corr = df
    if CORRELATION_SAMPLE_SIZE > 0 and len(df_corr) > CORRELATION_SAMPLE_SIZE:
        df_corr = df_corr.sample(CORRELATION_SAMPLE_SIZE, random_state=42)
    corr = df_corr[numeric_cols].corr()
    correlated = {}
    for col in numeric_cols:
        if col in numeric_required:
            continue
        best_feature = None
        best_corr = 0.0
        for req in numeric_required:
            if req not in corr.columns or col not in corr.index:
                continue
            corr_val = corr.at[col, req]
            if pd.isna(corr_val):
                continue
            if abs(corr_val) > abs(best_corr): # type: ignore
                best_corr = float(corr_val) # type: ignore
                best_feature = req
        if best_feature is None or abs(best_corr) < threshold:
            continue
        proxy_values = df_corr[best_feature].to_numpy()
        if np.nanstd(proxy_values) == 0:
            continue
        slope, intercept = np.polyfit(proxy_values, df_corr[col].to_numpy(), 1)
        correlated[col] = {
            "proxy": best_feature,
            "slope": float(slope),
            "intercept": float(intercept),
            "corr": float(best_corr),
        }
    return correlated


def _reduce_input_columns(preprocessor: PreprocessorArtifacts) -> list[str]:
    cols = [col for col in REDUCED_INPUT_FEATURES if col in preprocessor.input_feature_columns or col == "SK_ID_CURR"]
    if "SK_ID_CURR" not in cols:
        cols.append("SK_ID_CURR")
    return sorted(set(cols))


def _compute_correlated_imputation(
    data_path: Path,
    preprocessor: PreprocessorArtifacts,
) -> dict[str, dict[str, float | str]]:
    df = pd.read_parquet(data_path)
    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df[preprocessor.columns_keep]
    df = df.dropna(subset=preprocessor.columns_must_not_missing)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(pd.Series(preprocessor.numeric_medians))

    for col in preprocessor.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != "XNA"]

    for col, max_val in preprocessor.outlier_maxes.items():
        if col in df.columns:
            df = df[df[col] != max_val]

    return _build_correlated_imputation(
        df,
        input_feature_columns=preprocessor.input_feature_columns,
        numeric_required=preprocessor.numeric_required_columns,
        threshold=CORRELATION_THRESHOLD,
    )


def _ensure_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns or df[col].isna().any()]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required input columns.",
                "missing_columns": missing[:25],
                "missing_count": len(missing),
            },
        )


def _validate_numeric_inputs(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    invalid = []
    for col in numeric_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            invalid.append(col)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid numeric values provided.",
                "invalid_columns": invalid[:25],
                "invalid_count": len(invalid),
            },
        )


def _validate_numeric_ranges(df: pd.DataFrame, numeric_ranges: dict[str, tuple[float, float]]) -> None:
    if not numeric_ranges:
        return
    out_of_range = []
    for col, (min_val, max_val) in numeric_ranges.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            continue
        if ((values < min_val) | (values > max_val)).any():
            out_of_range.append(col)
    if out_of_range:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input contains values outside expected ranges.",
                "out_of_range_columns": out_of_range[:25],
                "out_of_range_count": len(out_of_range),
            },
        )


def _apply_correlated_imputation(df: pd.DataFrame, artifacts: PreprocessorArtifacts) -> None:
    correlated = getattr(artifacts, "correlated_imputation", {}) or {}
    if not correlated:
        return
    for col, info in correlated.items():
        if col not in df.columns or col in artifacts.required_input_columns:
            continue
        proxy = info.get("proxy")
        if proxy is None or proxy not in df.columns:
            continue
        missing = df[col].isna()
        if not missing.any():
            continue
        proxy_values = pd.to_numeric(df[proxy], errors="coerce")
        if proxy_values.isna().any():
            continue
        df.loc[missing, col] = info["slope"] * proxy_values[missing] + info["intercept"]
        if col in artifacts.numeric_ranges:
            min_val, max_val = artifacts.numeric_ranges[col]
            df.loc[missing, col] = df.loc[missing, col].clip(min_val, max_val)


def preprocess_input(df_raw: pd.DataFrame, artifacts: PreprocessorArtifacts) -> pd.DataFrame:
    df = df_raw.copy()

    for col in artifacts.required_input_columns:
        if col not in df.columns:
            df[col] = np.nan

    _ensure_required_columns(df, artifacts.required_input_columns)
    _validate_numeric_inputs(df, artifacts.numeric_required_columns)
    _validate_numeric_ranges(df, {k: v for k, v in artifacts.numeric_ranges.items() if k in artifacts.numeric_required_columns})

    df["is_train"] = 0
    df["is_test"] = 1
    if "TARGET" not in df.columns:
        df["TARGET"] = 0

    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in artifacts.columns_keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[artifacts.columns_keep]

    _apply_correlated_imputation(df, artifacts)

    for col, median in artifacts.numeric_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(median)

    for col in artifacts.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    _ensure_required_columns(df, artifacts.required_input_columns)

    if "CODE_GENDER" in df.columns and (df["CODE_GENDER"] == "XNA").any():
        raise HTTPException(
            status_code=422,
            detail={"message": "CODE_GENDER cannot be 'XNA' based on training rules."},
        )

    for col, max_val in artifacts.outlier_maxes.items():
        if col in df.columns and (df[col] >= max_val).any():
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Input contains outlier values removed during training.",
                    "outlier_columns": [col],
                },
            )

    df_hot = pd.get_dummies(df, columns=artifacts.categorical_columns)
    for col in artifacts.features_to_scaled:
        if col not in df_hot.columns:
            df_hot[col] = 0
    df_hot = df_hot[artifacts.features_to_scaled]

    scaled = artifacts.scaler.transform(df_hot)
    return pd.DataFrame(scaled, columns=artifacts.features_to_scaled, index=df.index)


@app.on_event("startup")
def startup_event() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    logger.info("Loading model from %s", MODEL_PATH)
    app.state.model = load_model(MODEL_PATH)

    logger.info("Loading preprocessor artifacts from %s", ARTIFACTS_PATH)
    app.state.preprocessor = load_preprocessor(DATA_PATH, ARTIFACTS_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Credit Scoring API. See /docs for Swagger UI."}


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/features")
def features(include_all: bool = Query(default=False)) -> dict[str, Any]:
    preprocessor: PreprocessorArtifacts = app.state.preprocessor
    optional_features = [col for col in preprocessor.input_feature_columns if col not in preprocessor.required_input_columns]
    correlated = sorted(getattr(preprocessor, "correlated_imputation", {}) or {})
    payload = {
        "required_input_features": preprocessor.required_input_columns,
        "engineered_features": ENGINEERED_FEATURES,
        "model_features_count": len(preprocessor.features_to_scaled),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "correlated_imputation_count": len(correlated),
        "correlated_imputation_features": correlated[:50],
    }
    if include_all:
        payload["input_features"] = preprocessor.input_feature_columns
        payload["optional_input_features"] = optional_features
    else:
        payload["input_features"] = preprocessor.required_input_columns
        payload["optional_input_features"] = []
        payload["optional_input_features_count"] = len(optional_features)
    return payload


@app.post("/predict")
def predict(
    payload: PredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    model = app.state.model
    preprocessor: PreprocessorArtifacts = app.state.preprocessor
    records = payload.data if isinstance(payload.data, list) else [payload.data]

    if not records:
        raise HTTPException(status_code=422, detail={"message": "No input records provided."})

    df_raw = pd.DataFrame.from_records(records)
    if "SK_ID_CURR" not in df_raw.columns:
        raise HTTPException(status_code=422, detail={"message": "SK_ID_CURR is required."})

    sk_ids = df_raw["SK_ID_CURR"].tolist()
    features = preprocess_input(df_raw, preprocessor)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[:, 1]
        use_threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        preds = (proba >= use_threshold).astype(int)
        results = [
            {
                "sk_id_curr": sk_id,
                "probability": float(prob),
                "prediction": int(pred),
            }
            for sk_id, prob, pred in zip(sk_ids, proba, preds)
        ]
        return {"predictions": results, "threshold": use_threshold}

    preds = model.predict(features)
    results = [
        {
            "sk_id_curr": sk_id,
            "prediction": int(pred),
        }
        for sk_id, pred in zip(sk_ids, preds)
    ]
    return {"predictions": results, "threshold": None}
