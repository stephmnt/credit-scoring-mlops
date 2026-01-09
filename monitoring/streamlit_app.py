from __future__ import annotations

from collections import Counter
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

from monitoring.drift_report import (
    CATEGORICAL_FEATURES,
    DAYS_EMPLOYED_SENTINEL,
    compute_drift_summary,
    generate_report,
    summarize_data_quality,
    summarize_errors,
    _load_logs,
    _prepare_categorical,
)


def _load_logs_safe(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not log_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return _load_logs(log_path)


def _filter_by_time(
    meta_df: pd.DataFrame,
    inputs_df: pd.DataFrame,
    since: str | None,
    until: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not since and not until:
        return meta_df, inputs_df, ""
    if "timestamp" not in meta_df.columns:
        return meta_df, inputs_df, "timestamp_missing"
    timestamps = pd.to_datetime(meta_df["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().all():
        return meta_df, inputs_df, "timestamp_invalid"
    mask = pd.Series(True, index=meta_df.index)
    if since:
        since_dt = pd.to_datetime(since, errors="coerce", utc=True)
        if not pd.isna(since_dt):
            mask &= timestamps >= since_dt
    if until:
        until_dt = pd.to_datetime(until, errors="coerce", utc=True)
        if not pd.isna(until_dt):
            mask &= timestamps <= until_dt
    return meta_df.loc[mask].reset_index(drop=True), inputs_df.loc[mask].reset_index(drop=True), "filtered"


def _count_dq_columns(meta_df: pd.DataFrame, key: str) -> Counter:
    counts: Counter = Counter()
    if "data_quality" not in meta_df.columns:
        return counts
    for item in meta_df["data_quality"].dropna():
        if not isinstance(item, dict):
            continue
        values = item.get(key)
        if isinstance(values, list):
            counts.update(values)
    return counts


def _counts_to_frame(counts: Counter, limit: int = 5) -> pd.DataFrame:
    if not counts:
        return pd.DataFrame()
    return pd.DataFrame(counts.most_common(limit), columns=["feature", "count"])


@st.cache_data(show_spinner=False)
def _cached_drift_summary(
    log_path: Path,
    reference_path: Path,
    sample_size: int,
    psi_threshold: float,
    score_bins: int,
    min_prod_samples: int,
    psi_eps: float,
    min_category_share: float,
    fdr_alpha: float,
    min_drift_features: int,
    prod_since: str | None,
    prod_until: str | None,
) -> dict[str, object]:
    return compute_drift_summary(
        log_path=log_path,
        reference_path=reference_path,
        sample_size=sample_size,
        psi_threshold=psi_threshold,
        score_bins=score_bins,
        min_prod_samples=min_prod_samples,
        psi_eps=psi_eps,
        min_category_share=min_category_share,
        fdr_alpha=fdr_alpha,
        min_drift_features=min_drift_features,
        prod_since=prod_since,
        prod_until=prod_until,
    )


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
    min_prod_samples = st.number_input("Min prod samples", min_value=10, max_value=5000, value=50, step=50)
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
    time_bucket = st.selectbox("Time bucket", ["1H", "6H", "1D"], index=2)
    show_preview = st.checkbox("Show log preview", value=False)
    preview_rows = st.number_input("Preview rows", min_value=10, max_value=1000, value=200, step=50)

inputs_df, meta_df = _load_logs_safe(log_path)

if meta_df.empty:
    st.warning("No logs found. Check the logs path.")
    st.stop()

meta_df, inputs_df, window_status = _filter_by_time(
    meta_df, inputs_df, prod_since or None, prod_until or None
)
if window_status in {"timestamp_missing", "timestamp_invalid"}:
    st.info(f"Time filter ignored ({window_status}).")

total_calls = len(meta_df)
valid_mask = meta_df.get("status_code", pd.Series(dtype=int)).fillna(0) < 400
valid_meta = meta_df.loc[valid_mask]
prod_inputs = inputs_df.loc[valid_mask] if not inputs_df.empty else inputs_df
success_rate = float(valid_mask.mean()) if total_calls else 0.0
error_rate = float((meta_df.get("status_code", pd.Series(dtype=int)) >= 400).mean()) if total_calls else 0.0
latency_ms = meta_df.get("latency_ms", pd.Series(dtype=float)).dropna()
latency_p50 = float(latency_ms.quantile(0.5)) if not latency_ms.empty else 0.0
latency_p95 = float(latency_ms.quantile(0.95)) if not latency_ms.empty else 0.0
latency_p99 = float(latency_ms.quantile(0.99)) if not latency_ms.empty else 0.0
latency_mean = float(latency_ms.mean()) if not latency_ms.empty else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total calls", f"{total_calls}")
col2.metric("Success rate", f"{success_rate:.2%}")
col3.metric("Error rate", f"{error_rate:.2%}")
col4.metric("Latency p50", f"{latency_p50:.2f} ms")
col5.metric("Latency p95", f"{latency_p95:.2f} ms")
st.caption(f"Latency p99: {latency_p99:.2f} ms | Mean: {latency_mean:.2f} ms")

st.subheader("Log Storage")
if log_path.exists():
    log_stat = log_path.stat()
    st.write(f"Path: `{log_path}`")
    st.write(f"Size: {log_stat.st_size / (1024 * 1024):.2f} MB")
    st.write(f"Last modified: {pd.to_datetime(log_stat.st_mtime, unit='s')}")
    if show_preview:
        st.dataframe(meta_df.tail(int(preview_rows)), use_container_width=True)
else:
    st.info("Log file not found.")

st.subheader("Traffic & Latency")
timestamps = pd.to_datetime(meta_df.get("timestamp", pd.Series(dtype=object)), errors="coerce", utc=True)
if not timestamps.isna().all():
    ts_df = meta_df.copy()
    ts_df["timestamp"] = timestamps
    ts_df = ts_df.dropna(subset=["timestamp"])
    if not ts_df.empty:
        calls_series = ts_df.set_index("timestamp").resample(time_bucket).size()
        st.line_chart(calls_series.rename("calls"))
        if "latency_ms" in ts_df.columns:
            latency_series = ts_df.set_index("timestamp")["latency_ms"].resample(time_bucket).median()
            st.line_chart(latency_series.rename("latency_p50_ms"))
else:
    st.info("No valid timestamps available for time series charts.")

if not latency_ms.empty:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(latency_ms, bins=30, color="#4C78A8", alpha=0.8)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency distribution")
    st.pyplot(fig, clear_figure=True)

scores = pd.to_numeric(valid_meta.get("probability", pd.Series(dtype=float)), errors="coerce").dropna()
predictions = pd.to_numeric(valid_meta.get("prediction", pd.Series(dtype=float)), errors="coerce").dropna()

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

if not valid_meta.empty and "timestamp" in valid_meta.columns and not scores.empty:
    score_ts = valid_meta.copy()
    score_ts["timestamp"] = pd.to_datetime(score_ts["timestamp"], errors="coerce", utc=True)
    score_ts["score"] = pd.to_numeric(score_ts.get("probability", pd.Series(dtype=float)), errors="coerce")
    score_ts = score_ts.dropna(subset=["timestamp", "score"])
    if not score_ts.empty:
        score_series = score_ts.set_index("timestamp")["score"].resample(time_bucket).mean()
        st.line_chart(score_series.rename("avg_score"))

st.subheader("Data Quality & Errors")
sentinel_rate = 0.0
if "DAYS_EMPLOYED" in prod_inputs.columns:
    sentinel_rate = float(
        (pd.to_numeric(prod_inputs["DAYS_EMPLOYED"], errors="coerce") == DAYS_EMPLOYED_SENTINEL).mean()
    )
dq_metrics = summarize_data_quality(meta_df, prod_inputs, {"production": sentinel_rate})
if dq_metrics.get("source") == "none":
    st.info("No data quality metrics available.")
else:
    dq_table = pd.DataFrame(
        [
            {"metric": "missing_required_rate", "value": dq_metrics.get("missing_required_rate", 0.0)},
            {"metric": "invalid_numeric_rate", "value": dq_metrics.get("invalid_numeric_rate", 0.0)},
            {"metric": "out_of_range_rate", "value": dq_metrics.get("out_of_range_rate", 0.0)},
            {"metric": "outlier_rate", "value": dq_metrics.get("outlier_rate", 0.0)},
            {"metric": "nan_rate", "value": dq_metrics.get("nan_rate", 0.0)},
            {"metric": "unknown_gender_rate", "value": dq_metrics.get("unknown_gender_rate", 0.0)},
            {"metric": "unknown_car_rate", "value": dq_metrics.get("unknown_car_rate", 0.0)},
            {"metric": "days_employed_sentinel_rate", "value": dq_metrics.get("days_employed_sentinel_rate", 0.0)},
        ]
    )
    dq_table["value"] = dq_table["value"].map(lambda v: f"{float(v):.2%}")
    st.table(dq_table)

issues = {
    "missing_required_columns": "Missing required",
    "invalid_numeric_columns": "Invalid numeric",
    "out_of_range_columns": "Out of range",
    "outlier_columns": "Outliers",
    "unknown_categories": "Unknown categories",
}
for key, label in issues.items():
    df_counts = _counts_to_frame(_count_dq_columns(meta_df, key))
    if not df_counts.empty:
        st.caption(label)
        st.dataframe(df_counts, hide_index=True, use_container_width=True)

error_breakdown = summarize_errors(meta_df[meta_df.get("status_code", pd.Series(dtype=int)) >= 400])
if error_breakdown:
    st.caption("Top error reasons")
    st.table(pd.DataFrame(error_breakdown, columns=["error", "count"]))

st.subheader("Data Drift")
if not reference_path.exists():
    st.warning("Reference dataset not found. Drift summary disabled.")
else:
    try:
        summary = _cached_drift_summary(
            log_path=log_path,
            reference_path=reference_path,
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
        summary_df = summary["summary_df"]
        n_prod = summary["n_prod"]
        n_ref = summary["n_ref"]
        drift_count = summary["drift_count"]
        drift_features = summary["drift_features"]
        if n_prod < int(min_prod_samples):
            st.warning("Sample insuffisant: drift non fiable (gate active).")
        st.metric("Drifted features", f"{drift_count}")
        if drift_features:
            st.write(f"Drifted: {', '.join(drift_features)}")

        show_only_drifted = st.checkbox("Show only drifted features", value=False)
        table_df = summary_df
        if show_only_drifted:
            table_df = summary_df[summary_df["drift_detected"] == True]
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        if not summary_df.empty:
            feature = st.selectbox("Feature to inspect", summary_df["feature"].tolist())
            row = summary_df.loc[summary_df["feature"] == feature].iloc[0]
            production_df = summary["production_df"]
            reference_df = summary["reference_df"]
            fig, ax = plt.subplots(figsize=(6, 3))
            if feature in CATEGORICAL_FEATURES:
                ref_series, prod_series = _prepare_categorical(
                    reference_df[feature],
                    production_df[feature],
                    min_share=float(min_category_share),
                    other_label="OTHER",
                )
                plot_df = pd.DataFrame(
                    {
                        "reference": ref_series.value_counts(normalize=True),
                        "production": prod_series.value_counts(normalize=True),
                    }
                ).fillna(0)
                plot_df.plot(kind="bar", ax=ax)
                ax.set_title(f"Distribution: {feature}")
                ax.set_ylabel("Share")
                psi_value = row.get("psi")
                if psi_value is not None:
                    st.caption(f"PSI: {psi_value} | n_prod: {row.get('n_prod')} | n_ref: {row.get('n_ref')}")
            else:
                ax.hist(reference_df[feature].dropna(), bins=30, alpha=0.6, label="reference")
                ax.hist(production_df[feature].dropna(), bins=30, alpha=0.6, label="production")
                ax.set_title(f"Distribution: {feature}")
                ax.legend()
                st.caption(
                    f"KS: {row.get('ks_stat')} | p_value: {row.get('p_value_fdr') or row.get('p_value')}"
                )
            st.pyplot(fig, clear_figure=True)

            if row.get("drift_detected"):
                st.warning("Drift detected: investigate data pipeline and model stability.")
            else:
                st.success("No drift signal for this feature.")
    except SystemExit as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(str(exc))

st.subheader("Generate Drift Report")
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
