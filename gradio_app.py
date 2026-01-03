from __future__ import annotations

from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.main import (
    MinimalPredictionRequest,
    app,
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
                "value": np.nan,
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


def _compute_shap_top_features(record: dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    preprocessor = app.state.preprocessor
    model = app.state.model
    df_raw = pd.DataFrame.from_records([record])
    df_norm, _, _ = _normalize_inputs(df_raw, preprocessor)
    features = prepare_inference_features(df_norm, preprocessor, model)
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

    explainer = getattr(app.state, "shap_explainer", None)
    if explainer is None:
        try:
            explainer = shap.TreeExplainer(estimator)
        except Exception:
            explainer = shap.Explainer(estimator, X_shap)
        app.state.shap_explainer = explainer

    try:
        explanation = explainer(X_shap)
        values = _extract_shap_values(explanation.values)
    except Exception:
        values = _extract_shap_values(explainer.shap_values(X_shap)) # type: ignore

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
            "value": float(feature_values[idx]),
            "shap_value": float(shap_row[idx]),
        }
        for idx in top_idx
    ]
    return pd.DataFrame(rows)


def score_minimal(
    sk_id_curr: float,
    amt_credit: float,
    duration_months: float,
    threshold: float,
) -> tuple[float | None, str, float | None, pd.DataFrame, dict[str, Any]]:
    _ensure_startup()
    try:
        payload = MinimalPredictionRequest(
            sk_id_curr=int(sk_id_curr),
            amt_credit=float(amt_credit),
            duration_months=int(duration_months),
        )
        record = _build_minimal_record(payload, app.state.preprocessor)
        response = predict_minimal(payload, threshold=float(threshold), x_client_source="gradio")
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
        return probability, label, float(response.get("threshold", 0.0)), shap_table, snapshot
    except HTTPException as exc:
        return None, f"Erreur: {exc.detail}", None, _shap_error_table("No SHAP available."), {"error": exc.detail}
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, f"Erreur: {exc}", None, _shap_error_table("No SHAP available."), {"error": str(exc)}


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
        threshold = gr.Slider(label="Seuil", minimum=0.0, maximum=1.0, value=0.5, step=0.01)

    run_btn = gr.Button("Scorer")

    with gr.Row():
        probability = gr.Number(label="Probabilité de défaut")
        prediction = gr.Textbox(label="Prédiction")
        threshold_used = gr.Number(label="Seuil utilisé")

    shap_table = gr.Dataframe(
        headers=["feature", "value", "shap_value"],
        label="Top 10 SHAP (local)",
        datatype=["str", "number", "number"],
        interactive=False,
    )

    snapshot = gr.JSON(label="Snapshot client (référence)")

    run_btn.click(
        score_minimal,
        inputs=[sk_id_curr, amt_credit, duration_months, threshold],
        outputs=[probability, prediction, threshold_used, shap_table, snapshot],
    )


if __name__ == "__main__":
    _ensure_startup()
    demo.launch()
