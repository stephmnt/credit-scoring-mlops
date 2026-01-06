from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.drift_report import generate_report, _load_logs


def _load_logs_safe(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not log_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return _load_logs(log_path)


st.set_page_config(page_title="Credit Scoring Monitoring", layout="wide")
st.title("Credit Scoring Monitoring")

with st.sidebar:
    st.header("Inputs")
    log_path = Path(st.text_input("Logs path", "logs/predictions.jsonl"))
    reference_path = Path(st.text_input("Reference data", "data/data_final.parquet"))
    output_dir = Path(st.text_input("Output dir", "reports"))
    sample_size = st.number_input("Sample size", min_value=1000, max_value=200000, value=50000, step=1000)
    psi_threshold = st.number_input("PSI threshold", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    score_bins = st.number_input("Score bins", min_value=10, max_value=100, value=30, step=5)
    min_prod_samples = st.number_input("Min prod samples", min_value=10, max_value=5000, value=200, step=50)
    psi_eps = st.number_input("PSI epsilon", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
    min_category_share = st.number_input(
        "Min category share",
        min_value=0.001,
        max_value=0.2,
        value=0.01,
        step=0.005,
        format="%.3f",
    )
    fdr_alpha = st.number_input("FDR alpha", min_value=0.01, max_value=0.2, value=0.05, step=0.01, format="%.2f")
    min_drift_features = st.number_input("Min drift features", min_value=1, max_value=10, value=1, step=1)
    prod_since = st.text_input("Prod since (ISO)", "")
    prod_until = st.text_input("Prod until (ISO)", "")

inputs_df, meta_df = _load_logs_safe(log_path)

if meta_df.empty:
    st.warning("No logs found. Check the logs path.")
    st.stop()

total_calls = len(meta_df)
valid_mask = meta_df.get("status_code", pd.Series(dtype=int)).fillna(0) < 400
prod_inputs = inputs_df.loc[valid_mask] if not inputs_df.empty else inputs_df
n_prod = len(prod_inputs)
error_rate = float((meta_df.get("status_code", pd.Series(dtype=int)) >= 400).mean()) if total_calls else 0.0
latency_ms = meta_df.get("latency_ms", pd.Series(dtype=float)).dropna()
latency_p50 = float(latency_ms.quantile(0.5)) if not latency_ms.empty else 0.0
latency_p95 = float(latency_ms.quantile(0.95)) if not latency_ms.empty else 0.0

valid_meta = meta_df
if "status_code" in meta_df.columns:
    valid_meta = meta_df[meta_df["status_code"] < 400]
scores = pd.to_numeric(valid_meta.get("probability", pd.Series(dtype=float)), errors="coerce").dropna()
predictions = pd.to_numeric(valid_meta.get("prediction", pd.Series(dtype=float)), errors="coerce").dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total calls", f"{total_calls}")
col2.metric("Error rate", f"{error_rate:.2%}")
col3.metric("Latency p50", f"{latency_p50:.2f} ms")
col4.metric("Latency p95", f"{latency_p95:.2f} ms")
st.caption(f"Production sample size (status < 400): {n_prod}")
if n_prod < int(min_prod_samples):
    st.warning("Sample insuffisant: drift non fiable (gate active).")

st.subheader("Score Monitoring")
if not scores.empty:
    score_stats = {
        "mean": float(scores.mean()),
        "p50": float(scores.quantile(0.5)),
        "p95": float(scores.quantile(0.95)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    st.json(score_stats)
    hist, bin_edges = np.histogram(scores, bins=int(score_bins), range=(0, 1))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", color="#4C78A8")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution")
    st.pyplot(fig, clear_figure=True)
else:
    st.info("No probability scores available in logs.")

if not predictions.empty:
    pred_rate = float(predictions.mean())
    st.metric("Predicted default rate", f"{pred_rate:.2%}")
    pred_counts = predictions.value_counts(normalize=True, dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(pred_counts.index.astype(str), pred_counts.values, color="#F58518")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction rate")
    st.pyplot(fig, clear_figure=True)

st.subheader("Data Drift")
if st.button("Generate drift report"):
    try:
        report_path = generate_report(
            log_path=log_path,
            reference_path=reference_path,
            output_dir=output_dir,
            sample_size=int(sample_size),
            psi_threshold=float(psi_threshold),
            score_bins=int(score_bins),
            min_prod_samples=int(min_prod_samples),
            psi_eps=float(psi_eps),
            min_category_share=float(min_category_share),
            fdr_alpha=float(fdr_alpha),
            min_drift_features=int(min_drift_features),
            prod_since=prod_since or None,
            prod_until=prod_until or None,
        )
        st.success(f"Generated: {report_path}")
    except ImportError as exc:
        st.error(
            "Parquet engine missing. Install `pyarrow` in this environment or run "
            "`python -m streamlit run monitoring/streamlit_app.py`."
        )
        st.exception(exc)

report_file = output_dir / "drift_report.html"
if report_file.exists():
    st.markdown(f"Report available at `{report_file}`")
