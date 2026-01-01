from __future__ import annotations

from typing import Any

import gradio as gr
from fastapi import HTTPException

from app.main import MinimalPredictionRequest, app, predict_minimal, startup_event


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


def score_minimal(
    sk_id_curr: float,
    amt_credit: float,
    duration_months: float,
    threshold: float,
) -> tuple[float | None, str, float | None, dict[str, Any]]:
    _ensure_startup()
    try:
        payload = MinimalPredictionRequest(
            sk_id_curr=int(sk_id_curr),
            amt_credit=float(amt_credit),
            duration_months=int(duration_months),
        )
        response = predict_minimal(payload, threshold=float(threshold))
        result = response["predictions"][0]
        probability = float(result.get("probability", 0.0))
        pred_value = int(result.get("prediction", 0))
        label = "Default (1)" if pred_value == 1 else "No default (0)"
        snapshot = _customer_snapshot(int(sk_id_curr))
        snapshot.update(
            {
                "AMT_CREDIT_REQUESTED": float(amt_credit),
                "DURATION_MONTHS": int(duration_months),
            }
        )
        return probability, label, float(response.get("threshold", 0.0)), snapshot
    except HTTPException as exc:
        return None, f"Erreur: {exc.detail}", None, {"error": exc.detail}
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, f"Erreur: {exc}", None, {"error": str(exc)}


with gr.Blocks(title="Credit Scoring - Minimal Inputs") as demo:
    gr.Markdown("# Credit Scoring - Minimal Inputs")
    gr.Markdown(
        "Renseignez l'identifiant client, le montant du credit et la duree. "
        "Les autres features proviennent des donnees clients reference."
    )

    with gr.Row():
        sk_id_curr = gr.Number(label="SK_ID_CURR", precision=0, value=100001)
        amt_credit = gr.Number(label="AMT_CREDIT", value=200000)
        duration_months = gr.Number(label="Duree (mois)", precision=0, value=60)
        threshold = gr.Slider(label="Seuil", minimum=0.0, maximum=1.0, value=0.5, step=0.01)

    run_btn = gr.Button("Scorer")

    with gr.Row():
        probability = gr.Number(label="Probabilite de defaut")
        prediction = gr.Textbox(label="Decision")
        threshold_used = gr.Number(label="Seuil utilise")

    snapshot = gr.JSON(label="Snapshot client (reference)")

    run_btn.click(
        score_minimal,
        inputs=[sk_id_curr, amt_credit, duration_months, threshold],
        outputs=[probability, prediction, threshold_used, snapshot],
    )


if __name__ == "__main__":
    _ensure_startup()
    demo.launch()
