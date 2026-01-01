from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import uuid
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Response
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
FEATURE_SELECTION_METHOD = os.getenv("FEATURE_SELECTION_METHOD", "correlation")
FEATURE_SELECTION_TOP_N = int(os.getenv("FEATURE_SELECTION_TOP_N", "8"))
FEATURE_SELECTION_MIN_CORR = float(os.getenv("FEATURE_SELECTION_MIN_CORR", "0.02"))
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.85"))
CORRELATION_SAMPLE_SIZE = int(os.getenv("CORRELATION_SAMPLE_SIZE", "50000"))
ALLOW_MISSING_ARTIFACTS = os.getenv("ALLOW_MISSING_ARTIFACTS", "0") == "1"
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "1") == "1"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = os.getenv("LOG_FILE", "predictions.jsonl")
LOG_INCLUDE_INPUTS = os.getenv("LOG_INCLUDE_INPUTS", "1") == "1"
LOG_HASH_SK_ID = os.getenv("LOG_HASH_SK_ID", "0") == "1"
MODEL_VERSION = os.getenv("MODEL_VERSION", MODEL_PATH.name)
LOGS_ACCESS_TOKEN = os.getenv("LOGS_ACCESS_TOKEN")
CUSTOMER_DATA_PATH = Path(os.getenv("CUSTOMER_DATA_PATH", str(DATA_PATH)))
CUSTOMER_LOOKUP_ENABLED = os.getenv("CUSTOMER_LOOKUP_ENABLED", "1") == "1"
CUSTOMER_LOOKUP_CACHE = os.getenv("CUSTOMER_LOOKUP_CACHE", "1") == "1"
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID")
HF_MODEL_REPO_TYPE = os.getenv("HF_MODEL_REPO_TYPE", "model")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", MODEL_PATH.name)
HF_PREPROCESSOR_REPO_ID = os.getenv("HF_PREPROCESSOR_REPO_ID", HF_MODEL_REPO_ID or "")
HF_PREPROCESSOR_REPO_TYPE = os.getenv("HF_PREPROCESSOR_REPO_TYPE", HF_MODEL_REPO_TYPE)
HF_PREPROCESSOR_FILENAME = os.getenv("HF_PREPROCESSOR_FILENAME", ARTIFACTS_PATH.name)
HF_CUSTOMER_REPO_ID = os.getenv("HF_CUSTOMER_REPO_ID")
HF_CUSTOMER_REPO_TYPE = os.getenv("HF_CUSTOMER_REPO_TYPE", "dataset")
HF_CUSTOMER_FILENAME = os.getenv("HF_CUSTOMER_FILENAME", CUSTOMER_DATA_PATH.name)

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
FEATURE_SELECTION_CATEGORICAL_INPUTS = ["CODE_GENDER", "FLAG_OWN_CAR"]
# Default reduced inputs (fallback when correlation-based selection is unavailable).
DEFAULT_REDUCED_INPUT_FEATURES = [
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

CODE_GENDER_MAPPING = {
    "F": "F",
    "FEMALE": "F",
    "0": "F",
    "W": "F",
    "WOMAN": "F",
    "M": "M",
    "MALE": "M",
    "1": "M",
    "MAN": "M",
}
FLAG_OWN_CAR_MAPPING = {
    "Y": "Y",
    "YES": "Y",
    "TRUE": "Y",
    "1": "Y",
    "T": "Y",
    "N": "N",
    "NO": "N",
    "FALSE": "N",
    "0": "N",
    "F": "N",
}
DAYS_EMPLOYED_SENTINEL = 365243


class PredictionRequest(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]]


class MinimalPredictionRequest(BaseModel):
    sk_id_curr: int
    amt_credit: float
    duration_months: int | None = None
    amt_annuity: float | None = None


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
    reduced_input_columns: list[str] = field(default_factory=list)
    feature_selection_method: str = "default"
    feature_selection_scores: dict[str, float] = field(default_factory=dict)


app = FastAPI(title="Credit Scoring API", version="0.1.0")


class DummyModel:
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        count = len(X)
        return np.tile([0.5, 0.5], (count, 1))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def _hash_value(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _normalize_category_value(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return np.nan
    key = str(value).strip().upper()
    if not key:
        return np.nan
    return mapping.get(key, "Unknown")


def _ensure_hf_asset(
    local_path: Path,
    repo_id: str | None,
    filename: str,
    repo_type: str,
) -> Path | None:
    if local_path.exists():
        return local_path
    if not repo_id:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required to download remote assets.") from exc
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
    )


def _normalize_inputs(
    df_raw: pd.DataFrame,
    preprocessor: PreprocessorArtifacts,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    df = df_raw.copy()
    for col in preprocessor.required_input_columns:
        if col not in df.columns:
            df[col] = np.nan

    unknown_masks: dict[str, pd.Series] = {}
    if "CODE_GENDER" in df.columns:
        raw = df["CODE_GENDER"]
        normalized = raw.apply(lambda v: _normalize_category_value(v, CODE_GENDER_MAPPING))
        unknown_masks["CODE_GENDER"] = normalized.eq("Unknown") & raw.notna()
        df["CODE_GENDER"] = normalized
    if "FLAG_OWN_CAR" in df.columns:
        raw = df["FLAG_OWN_CAR"]
        normalized = raw.apply(lambda v: _normalize_category_value(v, FLAG_OWN_CAR_MAPPING))
        unknown_masks["FLAG_OWN_CAR"] = normalized.eq("Unknown") & raw.notna()
        df["FLAG_OWN_CAR"] = normalized

    sentinel_mask = pd.Series(False, index=df.index)
    if "DAYS_EMPLOYED" in df.columns:
        values = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
        sentinel_mask = values == DAYS_EMPLOYED_SENTINEL
        if sentinel_mask.any():
            df.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan

    return df, unknown_masks, sentinel_mask


def _build_data_quality_records(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    unknown_masks: dict[str, pd.Series],
    sentinel_mask: pd.Series,
    preprocessor: PreprocessorArtifacts,
) -> list[dict[str, Any]]:
    required_cols = preprocessor.required_input_columns
    numeric_required = preprocessor.numeric_required_columns
    numeric_ranges = {
        col: bounds
        for col, bounds in preprocessor.numeric_ranges.items()
        if col in numeric_required
    }

    missing_mask = df_norm[required_cols].isna() if required_cols else pd.DataFrame(index=df_norm.index)
    invalid_masks: dict[str, pd.Series] = {}
    out_of_range_masks: dict[str, pd.Series] = {}

    for col in numeric_required:
        if col not in df_raw.columns:
            invalid_masks[col] = pd.Series(False, index=df_norm.index)
            continue
        raw = df_raw[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        invalid_masks[col] = coerced.isna() & raw.notna()

    for col, (min_val, max_val) in numeric_ranges.items():
        if col not in df_norm.columns:
            out_of_range_masks[col] = pd.Series(False, index=df_norm.index)
            continue
        values = pd.to_numeric(df_norm[col], errors="coerce")
        out_of_range_masks[col] = (values < min_val) | (values > max_val)

    records: list[dict[str, Any]] = []
    for idx in df_norm.index:
        missing_cols = (
            [col for col in required_cols if missing_mask.at[idx, col]]
            if required_cols
            else []
        )
        invalid_cols = [col for col, mask in invalid_masks.items() if mask.at[idx]]
        out_of_range_cols = [col for col, mask in out_of_range_masks.items() if mask.at[idx]]
        unknown_cols = [col for col, mask in unknown_masks.items() if mask.at[idx]]
        nan_rate = float(missing_mask.loc[idx].mean()) if not missing_mask.empty else 0.0
        records.append(
            {
                "missing_required_columns": missing_cols,
                "invalid_numeric_columns": invalid_cols,
                "out_of_range_columns": out_of_range_cols,
                "unknown_categories": unknown_cols,
                "days_employed_sentinel": bool(sentinel_mask.at[idx]) if not sentinel_mask.empty else False,
                "nan_rate": nan_rate,
            }
        )
    return records


def _build_minimal_record(
    payload: MinimalPredictionRequest,
    preprocessor: PreprocessorArtifacts,
) -> dict[str, Any]:
    reference = _get_customer_reference(preprocessor)
    if reference is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Customer reference data is not available."},
        )
    sk_id = int(payload.sk_id_curr)
    if sk_id not in reference.index:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Client {sk_id} not found in reference data."},
        )
    record = reference.loc[sk_id].to_dict()
    record["SK_ID_CURR"] = sk_id
    if payload.amt_credit <= 0:
        raise HTTPException(
            status_code=422,
            detail={"message": "AMT_CREDIT must be positive."},
        )
    record["AMT_CREDIT"] = float(payload.amt_credit)
    if payload.amt_annuity is not None:
        if payload.amt_annuity <= 0:
            raise HTTPException(
                status_code=422,
                detail={"message": "AMT_ANNUITY must be positive."},
            )
        record["AMT_ANNUITY"] = float(payload.amt_annuity)
    elif payload.duration_months is not None:
        if payload.duration_months <= 0:
            raise HTTPException(
                status_code=422,
                detail={"message": "duration_months must be positive."},
            )
        record["AMT_ANNUITY"] = float(payload.amt_credit) / float(payload.duration_months)
    else:
        raise HTTPException(
            status_code=422,
            detail={"message": "Provide duration_months or amt_annuity."},
        )
    if "AMT_GOODS_PRICE" in record:
        record["AMT_GOODS_PRICE"] = float(payload.amt_credit)
    return record


def _append_log_entries(entries: list[dict[str, Any]]) -> None:
    if not LOG_PREDICTIONS:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / LOG_FILE
        with log_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True, default=_json_fallback) + "\n")
    except OSError as exc:
        logger.warning("Failed to write prediction logs: %s", exc)


def _log_prediction_entries(
    request_id: str,
    records: list[dict[str, Any]],
    results: list[dict[str, Any]] | None,
    latency_ms: float,
    threshold: float | None,
    status_code: int,
    preprocessor: PreprocessorArtifacts,
    data_quality: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    if not LOG_PREDICTIONS:
        return
    if not records:
        records = [{}]
    timestamp = datetime.now(timezone.utc).isoformat()
    required_cols = preprocessor.required_input_columns
    entries: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        inputs: dict[str, Any] = {}
        if LOG_INCLUDE_INPUTS:
            inputs = {col: record.get(col) for col in required_cols if col in record}
            if LOG_HASH_SK_ID and "SK_ID_CURR" in inputs:
                inputs["SK_ID_CURR"] = _hash_value(inputs["SK_ID_CURR"])
        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "request_id": request_id,
            "endpoint": "/predict",
            "latency_ms": round(latency_ms, 3),
            "status_code": status_code,
            "model_version": MODEL_VERSION,
            "threshold": threshold,
            "inputs": inputs,
        }
        if data_quality and idx < len(data_quality):
            entry["data_quality"] = data_quality[idx]
        if results and idx < len(results):
            result = results[idx]
            sk_id = result.get("sk_id_curr")
            entry.update(
                {
                    "sk_id_curr": _hash_value(sk_id) if LOG_HASH_SK_ID and sk_id is not None else sk_id,
                    "probability": result.get("probability"),
                    "prediction": result.get("prediction"),
                }
            )
        if error:
            entry["error"] = error
        entries.append(entry)
    _append_log_entries(entries)


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

    reduced_input_columns, selection_scores, selection_method = _compute_reduced_inputs(
        df,
        input_feature_columns=input_feature_columns,
    )

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
        required_input = reduced_input_columns
        if not required_input:
            required_input = _fallback_reduced_inputs(input_feature_columns)
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
        reduced_input_columns=reduced_input_columns,
        feature_selection_method=selection_method,
        feature_selection_scores=selection_scores,
    )


def build_fallback_preprocessor() -> PreprocessorArtifacts:
    base = pd.DataFrame(
        [
            {
                "SK_ID_CURR": 100001,
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.61,
                "EXT_SOURCE_3": 0.75,
                "AMT_ANNUITY": 24700.5,
                "AMT_CREDIT": 406597.5,
                "AMT_GOODS_PRICE": 351000.0,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "AMT_INCOME_TOTAL": 202500.0,
                "CNT_FAM_MEMBERS": 1,
                "CNT_CHILDREN": 0,
            },
            {
                "SK_ID_CURR": 100002,
                "EXT_SOURCE_1": 0.35,
                "EXT_SOURCE_2": 0.52,
                "EXT_SOURCE_3": 0.68,
                "AMT_ANNUITY": 22000.0,
                "AMT_CREDIT": 350000.0,
                "AMT_GOODS_PRICE": 300000.0,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -1200,
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "Y",
                "AMT_INCOME_TOTAL": 180000.0,
                "CNT_FAM_MEMBERS": 2,
                "CNT_CHILDREN": 1,
            },
        ]
    )

    df = new_features_creation(base)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    columns_keep = df.columns.tolist()
    columns_must_not_missing = [col for col in columns_keep if col not in IGNORE_FEATURES]

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_medians = df[numeric_cols].median().to_dict()
    df[numeric_cols] = df[numeric_cols].fillna(numeric_medians)

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    df[categorical_columns] = df[categorical_columns].fillna("Unknown")

    df_hot = pd.get_dummies(df, columns=categorical_columns)
    features_to_scaled = [col for col in df_hot.columns if col not in IGNORE_FEATURES]
    scaler = MinMaxScaler()
    scaler.fit(df_hot[features_to_scaled])

    raw_feature_columns = df.columns.tolist()
    input_feature_columns = [c for c in raw_feature_columns if c not in ["is_train", "is_test", "TARGET"]]

    required_raw = set(ENGINEERED_SOURCES)
    required_raw.update(col for col in columns_must_not_missing if col in input_feature_columns)
    required_raw.add("SK_ID_CURR")
    required_input = _fallback_reduced_inputs(input_feature_columns)
    numeric_required = sorted(col for col in required_input if col in numeric_medians)

    numeric_ranges = {col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols}

    return PreprocessorArtifacts(
        columns_keep=columns_keep,
        columns_must_not_missing=columns_must_not_missing,
        numeric_medians={k: float(v) for k, v in numeric_medians.items()},
        categorical_columns=categorical_columns,
        outlier_maxes={},
        numeric_ranges=numeric_ranges,
        features_to_scaled=features_to_scaled,
        scaler=scaler,
        raw_feature_columns=raw_feature_columns,
        input_feature_columns=input_feature_columns,
        required_raw_columns=sorted(required_raw),
        required_input_columns=required_input,
        numeric_required_columns=numeric_required,
        correlated_imputation={},
        reduced_input_columns=required_input,
        feature_selection_method="fallback",
        feature_selection_scores={},
    )


def load_preprocessor(data_path: Path, artifacts_path: Path) -> PreprocessorArtifacts:
    if artifacts_path.exists():
        preprocessor = joblib.load(artifacts_path)
        updated = False
        required_updated = False
        if not hasattr(preprocessor, "reduced_input_columns") or not preprocessor.reduced_input_columns:
            reduced_cols, selection_scores, selection_method = _compute_reduced_inputs_from_data(
                data_path, preprocessor
            )
            preprocessor.reduced_input_columns = reduced_cols
            preprocessor.feature_selection_method = selection_method
            preprocessor.feature_selection_scores = selection_scores
            updated = True
        if not hasattr(preprocessor, "feature_selection_method"):
            preprocessor.feature_selection_method = "default"
            updated = True
        if not hasattr(preprocessor, "feature_selection_scores"):
            preprocessor.feature_selection_scores = {}
            updated = True
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


def _load_customer_reference(
    data_path: Path,
    preprocessor: PreprocessorArtifacts,
) -> pd.DataFrame:
    columns = list(preprocessor.input_feature_columns)
    if "SK_ID_CURR" not in columns:
        columns.insert(0, "SK_ID_CURR")
    df = pd.read_parquet(data_path, columns=columns)
    df = df.drop_duplicates(subset=["SK_ID_CURR"], keep="last").set_index("SK_ID_CURR")
    return df


def _get_customer_reference(preprocessor: PreprocessorArtifacts) -> pd.DataFrame | None:
    if not CUSTOMER_LOOKUP_ENABLED:
        return None
    cached = getattr(app.state, "customer_reference", None)
    if cached is not None:
        return cached
    data_path = CUSTOMER_DATA_PATH
    if not data_path.exists():
        downloaded = _ensure_hf_asset(
            data_path,
            HF_CUSTOMER_REPO_ID,
            HF_CUSTOMER_FILENAME,
            HF_CUSTOMER_REPO_TYPE,
        )
        if downloaded is None:
            return None
        data_path = downloaded
    ref = _load_customer_reference(data_path, preprocessor)
    if CUSTOMER_LOOKUP_CACHE:
        app.state.customer_reference = ref
    return ref


def _infer_numeric_ranges_from_scaler(preprocessor: PreprocessorArtifacts) -> dict[str, tuple[float, float]]:
    ranges = {}
    scaler = getattr(preprocessor, "scaler", None)
    if scaler is None or not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        return ranges
    for idx, col in enumerate(preprocessor.features_to_scaled):
        if col in preprocessor.numeric_medians:
            ranges[col] = (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
    return ranges


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _fallback_reduced_inputs(input_feature_columns: list[str]) -> list[str]:
    cols = [
        col
        for col in DEFAULT_REDUCED_INPUT_FEATURES
        if col in input_feature_columns or col == "SK_ID_CURR"
    ]
    if "SK_ID_CURR" not in cols:
        cols.insert(0, "SK_ID_CURR")
    return _dedupe_preserve_order(cols)


def _select_reduced_inputs_by_correlation(
    df: pd.DataFrame,
    *,
    input_feature_columns: list[str],
    top_n: int,
    min_corr: float,
) -> tuple[list[str], dict[str, float]]:
    if "TARGET" not in df.columns:
        return [], {}
    df_corr = df
    if CORRELATION_SAMPLE_SIZE > 0 and len(df_corr) > CORRELATION_SAMPLE_SIZE:
        df_corr = df_corr.sample(CORRELATION_SAMPLE_SIZE, random_state=42)
    numeric_cols = [
        col
        for col in df_corr.select_dtypes(include=["number"]).columns
        if col in input_feature_columns
        and col not in {"TARGET", "SK_ID_CURR", "is_train", "is_test"}
    ]
    if not numeric_cols:
        return [], {}
    corr = df_corr[numeric_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
    corr = corr.dropna()
    if corr.empty:
        return [], {}
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    if min_corr > 0:
        corr = corr[corr.abs() >= min_corr]
    selected_numeric = list(corr.index[:top_n])
    scores = {col: float(abs(corr.loc[col])) for col in selected_numeric}
    selected = ["SK_ID_CURR"]
    selected.extend(selected_numeric)
    selected.extend(
        col
        for col in FEATURE_SELECTION_CATEGORICAL_INPUTS
        if col in input_feature_columns
    )
    selected = [
        col for col in selected if col in input_feature_columns or col == "SK_ID_CURR"
    ]
    return _dedupe_preserve_order(selected), scores


def _compute_reduced_inputs(
    df: pd.DataFrame | None,
    *,
    input_feature_columns: list[str],
) -> tuple[list[str], dict[str, float], str]:
    if FEATURE_SELECTION_METHOD != "correlation":
        return _fallback_reduced_inputs(input_feature_columns), {}, "default"
    if df is None or "TARGET" not in df.columns:
        return _fallback_reduced_inputs(input_feature_columns), {}, "default"
    reduced_cols, scores = _select_reduced_inputs_by_correlation(
        df,
        input_feature_columns=input_feature_columns,
        top_n=FEATURE_SELECTION_TOP_N,
        min_corr=FEATURE_SELECTION_MIN_CORR,
    )
    if not reduced_cols:
        return _fallback_reduced_inputs(input_feature_columns), {}, "default"
    return reduced_cols, scores, "correlation"


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
    cols = getattr(preprocessor, "reduced_input_columns", None) or []
    if not cols:
        cols = _fallback_reduced_inputs(preprocessor.input_feature_columns)
    cols = [
        col
        for col in cols
        if col in preprocessor.input_feature_columns or col == "SK_ID_CURR"
    ]
    if "SK_ID_CURR" not in cols:
        cols.insert(0, "SK_ID_CURR")
    return _dedupe_preserve_order(cols)


def _compute_reduced_inputs_from_data(
    data_path: Path,
    preprocessor: PreprocessorArtifacts,
) -> tuple[list[str], dict[str, float], str]:
    if not data_path.exists():
        return _fallback_reduced_inputs(preprocessor.input_feature_columns), {}, "default"
    df = pd.read_parquet(data_path)
    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if preprocessor.columns_keep:
        df = df[preprocessor.columns_keep]
    if preprocessor.columns_must_not_missing:
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

    return _compute_reduced_inputs(df, input_feature_columns=preprocessor.input_feature_columns)


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


def _ensure_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    allow_missing: set[str] | None = None,
) -> None:
    allow_missing = allow_missing or set()
    missing = [
        col
        for col in required_cols
        if col not in df.columns or (col not in allow_missing and df[col].isna().any())
    ]
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
        if (coerced.isna() & df[col].notna()).any():
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
        mask = values.notna()
        if mask.any() and ((values[mask] < min_val) | (values[mask] > max_val)).any():
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

    allow_missing = {"DAYS_EMPLOYED"}
    _ensure_required_columns(df, artifacts.required_input_columns, allow_missing=allow_missing)
    _validate_numeric_inputs(df, artifacts.numeric_required_columns)
    _validate_numeric_ranges(df, {k: v for k, v in artifacts.numeric_ranges.items() if k in artifacts.numeric_required_columns})

    df["is_train"] = 0
    df["is_test"] = 1
    if "TARGET" not in df.columns:
        df["TARGET"] = 0

    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.reindex(columns=artifacts.columns_keep, fill_value=np.nan)

    _apply_correlated_imputation(df, artifacts)

    for col, median in artifacts.numeric_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(median)

    for col in artifacts.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    _ensure_required_columns(df, artifacts.required_input_columns, allow_missing=allow_missing)

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
    df_hot = df_hot.reindex(columns=artifacts.features_to_scaled, fill_value=0)

    scaled = artifacts.scaler.transform(df_hot)
    return pd.DataFrame(scaled, columns=artifacts.features_to_scaled, index=df.index)


@app.on_event("startup")
def startup_event() -> None:
    if getattr(app.state, "model", None) is not None and getattr(app.state, "preprocessor", None) is not None:
        return
    model_path = MODEL_PATH
    if not model_path.exists():
        downloaded = _ensure_hf_asset(
            model_path,
            HF_MODEL_REPO_ID,
            HF_MODEL_FILENAME,
            HF_MODEL_REPO_TYPE,
        )
        if downloaded is not None:
            model_path = downloaded
    if not model_path.exists():
        if ALLOW_MISSING_ARTIFACTS:
            logger.warning("Model file not found: %s. Using dummy model.", model_path)
            app.state.model = DummyModel()
        else:
            raise RuntimeError(f"Model file not found: {model_path}")
    else:
        logger.info("Loading model from %s", model_path)
        app.state.model = load_model(model_path)

    try:
        artifacts_path = ARTIFACTS_PATH
        if not artifacts_path.exists():
            downloaded = _ensure_hf_asset(
                artifacts_path,
                HF_PREPROCESSOR_REPO_ID or None,
                HF_PREPROCESSOR_FILENAME,
                HF_PREPROCESSOR_REPO_TYPE,
            )
            if downloaded is not None:
                artifacts_path = downloaded
        logger.info("Loading preprocessor artifacts from %s", artifacts_path)
        app.state.preprocessor = load_preprocessor(DATA_PATH, artifacts_path)
    except RuntimeError as exc:
        if ALLOW_MISSING_ARTIFACTS:
            logger.warning("Preprocessor artifacts missing (%s). Using fallback preprocessor.", exc)
            app.state.preprocessor = build_fallback_preprocessor()
        else:
            raise

    app.state.customer_reference = None
    if CUSTOMER_LOOKUP_ENABLED and CUSTOMER_LOOKUP_CACHE:
        try:
            ref = _get_customer_reference(app.state.preprocessor)
            if ref is not None:
                logger.info("Loaded customer reference data (%s rows)", len(ref))
            else:
                logger.warning("Customer reference data not available.")
        except Exception as exc:  # pragma: no cover - optional cache load
            logger.warning("Failed to load customer reference data: %s", exc)
    elif CUSTOMER_LOOKUP_ENABLED:
        logger.info("Customer lookup enabled without cache (on-demand load).")


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
    scores = getattr(preprocessor, "feature_selection_scores", {}) or {}
    selection_scores = {
        col: round(scores[col], 4)
        for col in preprocessor.required_input_columns
        if col in scores
    }
    payload = {
        "required_input_features": preprocessor.required_input_columns,
        "engineered_features": ENGINEERED_FEATURES,
        "model_features_count": len(preprocessor.features_to_scaled),
        "feature_selection_method": preprocessor.feature_selection_method,
        "feature_selection_top_n": FEATURE_SELECTION_TOP_N,
        "feature_selection_min_corr": FEATURE_SELECTION_MIN_CORR,
        "feature_selection_scores": selection_scores,
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


@app.get("/logs")
def logs(
    tail: int = Query(default=200, ge=1, le=2000),
    x_logs_token: str | None = Header(default=None, alias="X-Logs-Token"),
    authorization: str | None = Header(default=None),
) -> Response:
    if not LOGS_ACCESS_TOKEN:
        raise HTTPException(status_code=503, detail={"message": "Logs access token not configured."})

    token = x_logs_token
    if token is None and authorization:
        prefix = "bearer "
        if authorization.lower().startswith(prefix):
            token = authorization[len(prefix):].strip() or None

    if token != LOGS_ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail={"message": "Invalid logs access token."})

    if not LOG_PREDICTIONS:
        raise HTTPException(status_code=404, detail={"message": "Prediction logging is disabled."})

    log_path = LOG_DIR / LOG_FILE
    if not log_path.exists():
        raise HTTPException(status_code=404, detail={"message": "Log file not found."})

    with log_path.open("r", encoding="utf-8") as handle:
        lines = deque(handle, maxlen=tail)

    return Response(content="".join(lines), media_type="application/x-ndjson")


def _predict_records(records: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    model = app.state.model
    preprocessor: PreprocessorArtifacts = app.state.preprocessor
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    if not records:
        raise HTTPException(status_code=422, detail={"message": "No input records provided."})

    try:
        df_raw = pd.DataFrame.from_records(records)
        df_norm, unknown_masks, sentinel_mask = _normalize_inputs(df_raw, preprocessor)
        log_records = df_norm.to_dict(orient="records")
        dq_records = _build_data_quality_records(
            df_raw,
            df_norm,
            unknown_masks,
            sentinel_mask,
            preprocessor,
        )
        if "SK_ID_CURR" not in df_norm.columns:
            raise HTTPException(status_code=422, detail={"message": "SK_ID_CURR is required."})

        sk_ids = df_norm["SK_ID_CURR"].tolist()
        features = preprocess_input(df_norm, preprocessor)

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
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            _log_prediction_entries(
                request_id=request_id,
                records=log_records,
                results=results,
                latency_ms=latency_ms,
                threshold=use_threshold,
                status_code=200,
                preprocessor=preprocessor,
                data_quality=dq_records,
            )
            return {"predictions": results, "threshold": use_threshold}

        preds = model.predict(features)
        results = [
            {
                "sk_id_curr": sk_id,
                "prediction": int(pred),
            }
            for sk_id, pred in zip(sk_ids, preds)
        ]
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        _log_prediction_entries(
            request_id=request_id,
            records=log_records,
            results=results,
            latency_ms=latency_ms,
            threshold=None,
            status_code=200,
            preprocessor=preprocessor,
            data_quality=dq_records,
        )
        return {"predictions": results, "threshold": None}
    except HTTPException as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        _log_prediction_entries(
            request_id=request_id,
            records=log_records if "log_records" in locals() else records,
            results=None,
            latency_ms=latency_ms,
            threshold=threshold,
            status_code=exc.status_code,
            preprocessor=preprocessor,
            data_quality=dq_records if "dq_records" in locals() else None,
            error=json.dumps(detail, ensure_ascii=True),
        )
        raise
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        _log_prediction_entries(
            request_id=request_id,
            records=log_records if "log_records" in locals() else records,
            results=None,
            latency_ms=latency_ms,
            threshold=threshold,
            status_code=500,
            preprocessor=preprocessor,
            data_quality=dq_records if "dq_records" in locals() else None,
            error=str(exc),
        )
        raise


@app.post("/predict")
def predict(
    payload: PredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    records = payload.data if isinstance(payload.data, list) else [payload.data]
    return _predict_records(records, threshold)


@app.post("/predict-minimal")
def predict_minimal(
    payload: MinimalPredictionRequest,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    preprocessor: PreprocessorArtifacts = app.state.preprocessor
    record = _build_minimal_record(payload, preprocessor)
    return _predict_records([record], threshold)
