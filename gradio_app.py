from __future__ import annotations

from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.main import (
    DAYS_EMPLOYED_SENTINEL,
    ENGINEERED_SOURCES,
    MODEL_VERSION,
    MinimalPredictionRequest,
    app,
    new_features_creation,
    prepare_inference_features,
    predict_minimal,
    startup_event,
    _build_minimal_record,
    _normalize_inputs,
)


def _ensure_startup() -> None:
    if not getattr(app.state, "preprocessor", None):
        startup_event()


def _customer_snapshot(sk_id_curr: int) -> dict[str, Any]:
    reference = getattr(app.state, "customer_reference", None)
    if reference is None or sk_id_curr not in reference.index:
        return {}
    row = reference.loc[sk_id_curr]
    snapshot: dict[str, Any] = {"SK_ID_CURR": int(sk_id_curr)}
    if "CODE_GENDER" in row:
        snapshot["CODE_GENDER"] = row["CODE_GENDER"]
    if "FLAG_OWN_CAR" in row:
        snapshot["FLAG_OWN_CAR"] = row["FLAG_OWN_CAR"]
    if "AMT_INCOME_TOTAL" in row:
        snapshot["AMT_INCOME_TOTAL"] = float(row["AMT_INCOME_TOTAL"])
    if "DAYS_BIRTH" in row:
        snapshot["AGE_YEARS"] = round(abs(float(row["DAYS_BIRTH"])) / 365.25, 1)
    return snapshot


def _shap_error_table(message: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": message,
                "raw_value": np.nan,
                "shap_value": np.nan,
            }
        ]
    )


def _extract_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _clean_raw_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _strip_feature_prefix(feature_name: str) -> str:
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def _lookup_raw_value(feature_name: str, raw_df: pd.DataFrame, preprocessor) -> Any:
    cleaned_name = _strip_feature_prefix(feature_name)
    if cleaned_name in raw_df.columns:
        return raw_df.at[0, cleaned_name]
    for prefix in ("is_missing_", "is_outlier_"):
        if cleaned_name.startswith(prefix):
            base = cleaned_name[len(prefix):]
            if base in raw_df.columns:
                return raw_df.at[0, base]
    for col in getattr(preprocessor, "categorical_columns", []):
        if cleaned_name.startswith(f"{col}_") and col in raw_df.columns:
            return raw_df.at[0, col]
    return None

def _align_features_to_model(X: Any, model: Any) -> Any:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return X
    if isinstance(X, pd.DataFrame):
        return X.reindex(columns=list(expected), fill_value=0)
    return X

def _model_family(model: Any) -> str:
    name = type(model).__name__.lower()
    if "xgb" in name:
        return "xgb"
    if "lgbm" in name or "lightgbm" in name:
        return "lgbm"
    if "histgradientboosting" in name:
        return "histgb"
    return "unknown"

def _xgb_pred_contribs(estimator: Any, X: Any) -> np.ndarray:
    import xgboost as xgb

    if isinstance(X, pd.DataFrame):
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
    else:
        dm = xgb.DMatrix(np.asarray(X))

    booster = estimator.get_booster() if hasattr(estimator, "get_booster") else estimator
    contrib = booster.predict(dm, pred_contribs=True)
    return np.asarray(contrib)[:, :-1]


def _lgbm_pred_contribs(estimator: Any, X: Any) -> np.ndarray:
    contrib = estimator.predict(X, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


def _compute_shap_top_features(record: dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    preprocessor = app.state.preprocessor
    model = app.state.model
    df_raw = pd.DataFrame.from_records([record])
    df_norm, _, _ = _normalize_inputs(df_raw, preprocessor)
    raw_reference = new_features_creation(
        df_norm,
        days_employed_sentinel=DAYS_EMPLOYED_SENTINEL,
        engineered_sources=ENGINEERED_SOURCES,
    )
    features = prepare_inference_features(df_norm, preprocessor, model)
    features = _align_features_to_model(features, model)

    try:
        import shap
    except ImportError:
        return _shap_error_table("SHAP not installed.")

    estimator = model
    X_shap = features
    if hasattr(model, "named_steps") and model.named_steps.get("preprocessing") is not None:
        estimator = model.named_steps.get("estimator", model)
        pipeline_preprocessor = model.named_steps["preprocessing"]
        try:
            X_shap = pipeline_preprocessor.transform(features)
        except Exception as exc:
            return _shap_error_table(f"SHAP preprocessing failed: {exc}")
        try:
            import scipy.sparse as sp
            if sp.issparse(X_shap):
                X_shap = X_shap.toarray()
        except Exception:
            pass
        try:
            feature_names = pipeline_preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None
        if feature_names is not None:
            X_shap = pd.DataFrame(X_shap, columns=feature_names)

    family = _model_family(estimator)

    values: np.ndarray | None = None

    # 1) Contributions natives (meilleur choix pour XGB/LGBM)
    try:
        if family == "xgb":
            values = _xgb_pred_contribs(estimator, X_shap)
        elif family == "lgbm":
            values = _lgbm_pred_contribs(estimator, X_shap)
    except Exception:
        values = None

    # 2) Fallback SHAP (utile surtout pour HistGB / inconnus)
    if values is None:
        cache = getattr(app.state, "shap_explainer_cache", {})
        key = f"{MODEL_VERSION}:{type(estimator).__name__}"
        explainer = cache.get(key)

        if explainer is None:
            try:
                import shap
                predict_fn = (
                    (lambda X: estimator.predict_proba(X)[:, 1])
                    if hasattr(estimator, "predict_proba")
                    else (lambda X: estimator.predict(X))
                )

                # Evite le background dégénéré (1 seule ligne)
                if isinstance(X_shap, pd.DataFrame):
                    bg = pd.concat([X_shap] * 50, ignore_index=True)
                else:
                    bg = np.repeat(np.asarray(X_shap), repeats=50, axis=0)

                explainer = shap.Explainer(predict_fn, bg)
            except Exception as exc:
                return _shap_error_table(f"SHAP explainer init failed: {exc}")

            cache[key] = explainer
            app.state.shap_explainer_cache = cache

        try:
            import shap
            explanation = explainer(X_shap)
            values = _extract_shap_values(explanation.values)
        except Exception as exc:
            return _shap_error_table(f"SHAP failed: {exc}")

    shap_row = values[0]
    if isinstance(X_shap, pd.DataFrame):
        feature_values = X_shap.iloc[0].to_numpy()
        feature_names = X_shap.columns
    else:
        feature_values = np.asarray(X_shap)[0]
        feature_names = [f"feature_{idx}" for idx in range(len(feature_values))]
    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_k]
    rows = [
        {
            "feature": str(feature_names[idx]),
            "raw_value": _clean_raw_value(
                _lookup_raw_value(str(feature_names[idx]), raw_reference, preprocessor)
            ),
            "shap_value": float(np.round(shap_row[idx], 6)),
        }
        for idx in top_idx
    ]
    return pd.DataFrame(rows)


def score_minimal(
    sk_id_curr: float,
    amt_credit: float,
    duration_months: float,
) -> tuple[float | None, str, pd.DataFrame, dict[str, Any]]:
    _ensure_startup()
    try:
        payload = MinimalPredictionRequest(
            sk_id_curr=int(sk_id_curr),
            amt_credit=float(amt_credit),
            duration_months=int(duration_months),
        )
        record = _build_minimal_record(payload, app.state.preprocessor)
        response = predict_minimal(payload, threshold=None, x_client_source="gradio")
        result = response["predictions"][0]
        probability = float(result.get("probability", 0.0))
        pred_value = int(result.get("prediction", 0))
        label = "Default (1)" if pred_value == 1 else "No default (0)"
        shap_table = _compute_shap_top_features(record, top_k=10)
        snapshot = _customer_snapshot(int(sk_id_curr))
        snapshot.update(
            {
                "AMT_CREDIT_REQUESTED": float(amt_credit),
                "DURATION_MONTHS": int(duration_months),
            }
        )
        return probability, label, shap_table, snapshot
    except HTTPException as exc:
        return None, f"Erreur: {exc.detail}", _shap_error_table("No SHAP available."), {"error": exc.detail}
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, f"Erreur: {exc}", _shap_error_table("No SHAP available."), {"error": str(exc)}


with gr.Blocks(title="Credit scoring MLOps") as demo:
    gr.Markdown("# Credit scoring MLOps")
    gr.HTML("""
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
            <a href="https://github.com/stephmnt/credit-scoring-mlops/releases" target="_blank" rel="noreferrer">
                <img src="https://img.shields.io/github/v/release/stephmnt/credit-scoring-mlops" alt="GitHub Release" />
            </a>
            <a href="https://github.com/stephmnt/credit-scoring-mlops/actions/workflows/deploy.yml" target="_blank" rel="noreferrer">
                <img src="https://img.shields.io/github/actions/workflow/status/stephmnt/credit-scoring-mlops/deploy.yml" alt="GitHub Actions Workflow Status" />
            </a>
        </div>
        """)
    gr.Markdown(
        "Renseignez l'identifiant client, le montant du crédit et la durée. "
    )

    with gr.Row():
        sk_id_curr = gr.Number(label="Identifiant client", precision=0, value=100001)
        amt_credit = gr.Number(label="Montant du crédit", value=200000)
        duration_months = gr.Number(label="Durée (mois)", precision=0, value=60)

    run_btn = gr.Button("Scorer")

    with gr.Row():
        probability = gr.Number(label="Probabilité de défaut")
        prediction = gr.Textbox(label="Prédiction")

    shap_table = gr.Dataframe(
        headers=["feature", "raw_value", "shap_value"],
        label="Top 10 SHAP (local)",
        datatype=["str", "str", "number"],
        interactive=False,
    )

    snapshot = gr.JSON(label="Snapshot client (référence)")

    run_btn.click(
        score_minimal,
        inputs=[sk_id_curr, amt_credit, duration_months],
        outputs=[probability, prediction, shap_table, snapshot],
    )


if __name__ == "__main__":
    _ensure_startup()
    demo.launch()
