# construire drift avec evidently

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional plotting dependency
    raise SystemExit(
        "matplotlib is required for plots. Install it with: pip install matplotlib"
    ) from exc


DEFAULT_FEATURES = [
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

CATEGORICAL_FEATURES = {"CODE_GENDER", "FLAG_OWN_CAR"}
MIN_PROD_SAMPLES_DEFAULT = 200
PSI_EPS_DEFAULT = 1e-4
RARE_CATEGORY_MIN_SHARE_DEFAULT = 0.01
FDR_ALPHA_DEFAULT = 0.05
DAYS_EMPLOYED_SENTINEL = 365243

CATEGORY_NORMALIZATION = {
    "CODE_GENDER": {
        "F": "F",
        "FEMALE": "F",
        "0": "F",
        "W": "F",
        "WOMAN": "F",
        "M": "M",
        "MALE": "M",
        "1": "M",
        "MAN": "M",
    },
    "FLAG_OWN_CAR": {
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
    },
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)


def _load_logs(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        return pd.DataFrame(), pd.DataFrame()
    inputs = [
        entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
        for entry in entries
    ]
    inputs_df = pd.DataFrame.from_records(inputs)
    meta_df = pd.DataFrame.from_records(entries)
    return inputs_df, meta_df


def _normalize_category_value(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return np.nan
    key = str(value).strip().upper()
    if not key:
        return np.nan
    return mapping.get(key, "Unknown")


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature, mapping in CATEGORY_NORMALIZATION.items():
        if feature in out.columns:
            out[feature] = out[feature].apply(lambda v: _normalize_category_value(v, mapping))
    return out


def _replace_sentinel(series: pd.Series, sentinel: float) -> tuple[pd.Series, float]:
    values = pd.to_numeric(series, errors="coerce")
    sentinel_mask = values == sentinel
    if sentinel_mask.any():
        series = series.copy()
        series[sentinel_mask] = np.nan
    return series, float(sentinel_mask.mean()) if len(values) else 0.0


def _prepare_categorical(
    reference: pd.Series,
    production: pd.Series,
    min_share: float,
    max_categories: int | None = None,
    other_label: str = "__OTHER__",
) -> tuple[pd.Series, pd.Series]:
    ref_series = reference.fillna("Unknown")
    prod_series = production.fillna("Unknown")
    ref_freq = ref_series.value_counts(normalize=True)
    keep = ref_freq[ref_freq >= min_share].index.tolist()
    if max_categories is not None:
        keep = keep[:max_categories]
    ref_series = ref_series.where(ref_series.isin(keep), other=other_label)
    prod_series = prod_series.where(prod_series.isin(keep), other=other_label)
    return ref_series, prod_series


def _psi(reference: pd.Series, production: pd.Series, eps: float = PSI_EPS_DEFAULT) -> float:
    ref_freq = reference.value_counts(normalize=True, dropna=False)
    prod_freq = production.value_counts(normalize=True, dropna=False)
    categories = ref_freq.index.union(prod_freq.index)
    ref_probs = ref_freq.reindex(categories, fill_value=0).to_numpy()
    prod_probs = prod_freq.reindex(categories, fill_value=0).to_numpy()
    ref_probs = np.clip(ref_probs, eps, None)
    prod_probs = np.clip(prod_probs, eps, None)
    return float(np.sum((ref_probs - prod_probs) * np.log(ref_probs / prod_probs)))


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _plot_numeric(ref: pd.Series, prod: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(ref.dropna(), bins=30, alpha=0.6, label="reference")
    plt.hist(prod.dropna(), bins=30, alpha=0.6, label="production")
    plt.title(f"Distribution: {ref.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_categorical(ref: pd.Series, prod: pd.Series, output_path: Path) -> None:
    ref_freq = ref.value_counts(normalize=True)
    prod_freq = prod.value_counts(normalize=True)
    plot_df = pd.DataFrame({"reference": ref_freq, "production": prod_freq}).fillna(0)
    plot_df.sort_values("reference", ascending=False).plot(kind="bar", figsize=(7, 4))
    plt.title(f"Distribution: {ref.name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _benjamini_hochberg(pvalues: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    if not pvalues:
        return [], []
    pvals = np.array(pvalues, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    reject = np.zeros(m, dtype=bool)
    if below.any():
        cutoff = np.max(np.where(below)[0])
        reject[order[:cutoff + 1]] = True
    qvals = ranked * m / np.arange(1, m + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    adjusted = np.empty_like(qvals)
    adjusted[order] = qvals
    return adjusted.tolist(), reject.tolist()


def _extract_data_quality(meta_df: pd.DataFrame) -> list[dict[str, object]]:
    if "data_quality" not in meta_df.columns:
        return []
    dq_entries = []
    for item in meta_df["data_quality"].dropna():
        if isinstance(item, dict):
            dq_entries.append(item)
    return dq_entries


def _normalize_error_message(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        message = value.get("message")
        return str(message) if message else json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return ""
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
        return _normalize_error_message(parsed)
    return str(value)


def _summarize_errors(meta_df: pd.DataFrame, max_items: int = 5) -> list[tuple[str, int]]:
    if "error" not in meta_df.columns:
        return []
    errors = meta_df["error"].dropna().apply(_normalize_error_message)
    errors = errors[errors != ""]
    if errors.empty:
        return []
    counts = errors.value_counts().head(max_items)
    return list(zip(counts.index.tolist(), counts.tolist()))


def _dq_has_unknown(dq: dict[str, object], feature: str) -> bool:
    unknown = dq.get("unknown_categories")
    if isinstance(unknown, dict):
        return feature in unknown
    if isinstance(unknown, list):
        return feature in unknown
    return False


def _summarize_data_quality(
    meta_df: pd.DataFrame,
    production_df: pd.DataFrame,
    sentinel_rates: dict[str, float],
) -> dict[str, object]:
    dq_entries = _extract_data_quality(meta_df)
    if dq_entries:
        total = len(dq_entries)
        missing_rate = np.mean(
            [bool(dq.get("missing_required_columns")) for dq in dq_entries]
        )
        invalid_rate = np.mean(
            [bool(dq.get("invalid_numeric_columns")) for dq in dq_entries]
        )
        out_of_range_rate = np.mean(
            [bool(dq.get("out_of_range_columns")) for dq in dq_entries]
        )
        outlier_rate = np.mean(
            [bool(dq.get("outlier_columns")) for dq in dq_entries]
        )
        nan_rate = np.mean([float(dq.get("nan_rate", 0.0)) for dq in dq_entries])
        unknown_gender = np.mean(
            [_dq_has_unknown(dq, "CODE_GENDER") for dq in dq_entries]
        )
        unknown_car = np.mean(
            [_dq_has_unknown(dq, "FLAG_OWN_CAR") for dq in dq_entries]
        )
        sentinel_rate = np.mean(
            [bool(dq.get("days_employed_sentinel")) for dq in dq_entries]
        )
        return {
            "source": "log",
            "sample_size": total,
            "missing_required_rate": float(missing_rate),
            "invalid_numeric_rate": float(invalid_rate),
            "out_of_range_rate": float(out_of_range_rate),
            "outlier_rate": float(outlier_rate),
            "nan_rate": float(nan_rate),
            "unknown_gender_rate": float(unknown_gender),
            "unknown_car_rate": float(unknown_car),
            "days_employed_sentinel_rate": float(sentinel_rate),
        }

    if production_df.empty:
        return {"source": "none"}

    missing_rate = float(production_df.isna().any(axis=1).mean())
    unknown_gender_rate = 0.0
    unknown_car_rate = 0.0
    if "CODE_GENDER" in production_df.columns:
        unknown_gender_rate = float(
            (production_df["CODE_GENDER"] == "Unknown").mean()
        )
    if "FLAG_OWN_CAR" in production_df.columns:
        unknown_car_rate = float((production_df["FLAG_OWN_CAR"] == "Unknown").mean())
    sentinel_rate = float(sentinel_rates.get("production", 0.0))
    return {
        "source": "fallback",
        "sample_size": len(production_df),
        "missing_required_rate": missing_rate,
        "unknown_gender_rate": unknown_gender_rate,
        "unknown_car_rate": unknown_car_rate,
        "days_employed_sentinel_rate": sentinel_rate,
    }


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


def _plot_score_distribution(scores: pd.Series, output_path: Path, bins: int = 30) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(scores.dropna(), bins=bins, range=(0, 1), alpha=0.8, color="#4C78A8")
    plt.title("Prediction score distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_prediction_rate(predictions: pd.Series, output_path: Path) -> None:
    counts = predictions.value_counts(normalize=True, dropna=False).sort_index()
    plt.figure(figsize=(4, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#F58518")
    plt.title("Prediction rate")
    plt.xlabel("Predicted class")
    plt.ylabel("Share")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(
    log_path: Path,
    reference_path: Path,
    output_dir: Path,
    sample_size: int,
    psi_threshold: float,
    score_bins: int,
    min_prod_samples: int = MIN_PROD_SAMPLES_DEFAULT,
    psi_eps: float = PSI_EPS_DEFAULT,
    min_category_share: float = RARE_CATEGORY_MIN_SHARE_DEFAULT,
    fdr_alpha: float = FDR_ALPHA_DEFAULT,
    min_drift_features: int = 1,
    prod_since: str | None = None,
    prod_until: str | None = None,
) -> Path:
    inputs_df, meta_df = _load_logs(log_path)
    if meta_df.empty:
        raise SystemExit(f"No inputs found in logs: {log_path}")

    meta_df, inputs_df, window_status = _filter_by_time(
        meta_df, inputs_df, since=prod_since, until=prod_until
    )
    meta_df_all = meta_df.copy()
    inputs_df_all = inputs_df.copy()
    valid_mask = pd.Series(True, index=meta_df.index)
    if "status_code" in meta_df.columns:
        valid_mask = meta_df["status_code"].fillna(0) < 400
    inputs_df = inputs_df.loc[valid_mask].reset_index(drop=True)
    meta_df_valid = meta_df.loc[valid_mask].reset_index(drop=True)

    if inputs_df.empty:
        raise SystemExit(f"No valid inputs found in logs: {log_path}")

    features = [col for col in DEFAULT_FEATURES if col in inputs_df.columns]
    if not features:
        raise SystemExit("No matching features found in production logs.")

    reference_df = pd.read_parquet(reference_path, columns=features)
    if sample_size and len(reference_df) > sample_size:
        reference_df = reference_df.sample(sample_size, random_state=42)

    numeric_features = [col for col in features if col not in CATEGORICAL_FEATURES]
    production_df = _normalize_categories(inputs_df)
    reference_df = _normalize_categories(reference_df)
    production_df = _coerce_numeric(production_df, numeric_features)
    reference_df = _coerce_numeric(reference_df, numeric_features)

    sentinel_rates = {}
    if "DAYS_EMPLOYED" in production_df.columns:
        production_df["DAYS_EMPLOYED"], prod_rate = _replace_sentinel(
            production_df["DAYS_EMPLOYED"], DAYS_EMPLOYED_SENTINEL
        )
        reference_df["DAYS_EMPLOYED"], ref_rate = _replace_sentinel(
            reference_df["DAYS_EMPLOYED"], DAYS_EMPLOYED_SENTINEL
        )
        sentinel_rates = {
            "production": prod_rate,
            "reference": ref_rate,
        }

    summary_rows: list[dict[str, object]] = []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_prod = len(production_df)
    n_ref = len(reference_df)

    for feature in features:
        if feature not in reference_df.columns:
            continue
        ref_series = reference_df[feature]
        prod_series = production_df[feature]
        if feature in CATEGORICAL_FEATURES:
            feature_n_prod = int(prod_series.dropna().shape[0])
            feature_n_ref = int(ref_series.dropna().shape[0])
            ref_series, prod_series = _prepare_categorical(
                ref_series, prod_series, min_share=min_category_share, other_label="OTHER"
            )
            insufficient_sample = feature_n_prod < min_prod_samples
            psi_value = None
            if not insufficient_sample:
                psi_value = _psi(ref_series, prod_series, eps=psi_eps)
            summary_rows.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "psi": round(psi_value, 4) if psi_value is not None else None,
                    "drift_detected": bool(psi_value is not None and psi_value >= psi_threshold),
                    "n_prod": feature_n_prod,
                    "n_ref": feature_n_ref,
                    "note": "insufficient_sample" if insufficient_sample else "",
                }
            )
            plot_path = plots_dir / f"{_safe_name(feature)}.png"
            _plot_categorical(ref_series, prod_series, plot_path)
        else:
            ref_clean = ref_series.dropna()
            prod_clean = prod_series.dropna()
            if ref_clean.empty or prod_clean.empty:
                continue
            feature_n_prod = int(len(prod_clean))
            insufficient_sample = feature_n_prod < min_prod_samples
            stat = None
            pvalue = None
            if not insufficient_sample:
                stat, pvalue = stats.ks_2samp(ref_clean, prod_clean)
            summary_rows.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "ks_stat": round(float(stat), 4) if stat is not None else None,
                    "p_value": round(float(pvalue), 6) if pvalue is not None else None,
                    "p_value_fdr": None,
                    "drift_detected": bool(pvalue is not None and pvalue < 0.05),
                    "n_prod": feature_n_prod,
                    "n_ref": int(len(ref_clean)),
                    "note": "insufficient_sample" if insufficient_sample else "",
                }
            )
            plot_path = plots_dir / f"{_safe_name(feature)}.png"
            _plot_numeric(ref_series, prod_series, plot_path)

    numeric_rows = [
        (idx, row)
        for idx, row in enumerate(summary_rows)
        if row.get("type") == "numeric" and row.get("p_value") is not None
    ]
    if numeric_rows:
        pvalues = [row["p_value"] for _, row in numeric_rows]
        qvals, reject = _benjamini_hochberg(pvalues, alpha=fdr_alpha)
        for (idx, _), qval, rejected in zip(numeric_rows, qvals, reject):
            summary_rows[idx]["p_value_fdr"] = round(float(qval), 6)
            summary_rows[idx]["drift_detected"] = bool(rejected)

    summary_df = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.html"

    total_calls = len(meta_df_all)
    error_series = meta_df_all.get("status_code", pd.Series(dtype=int))
    error_rate = float((error_series >= 400).mean()) if total_calls else 0.0
    latency_ms = meta_df_all.get("latency_ms", pd.Series(dtype=float)).dropna()
    latency_p50 = float(latency_ms.quantile(0.5)) if not latency_ms.empty else 0.0
    latency_p95 = float(latency_ms.quantile(0.95)) if not latency_ms.empty else 0.0
    calls_with_inputs = int(inputs_df_all.notna().any(axis=1).sum()) if not inputs_df_all.empty else 0
    calls_with_dq = int(meta_df_all.get("data_quality", pd.Series(dtype=object)).notna().sum()) if total_calls else 0
    calls_success = int(valid_mask.sum())

    valid_meta = meta_df_valid
    score_series = (
        pd.to_numeric(valid_meta.get("probability", pd.Series(dtype=float)), errors="coerce")
        .dropna()
    )
    pred_series = (
        pd.to_numeric(valid_meta.get("prediction", pd.Series(dtype=float)), errors="coerce")
        .dropna()
    )

    score_metrics_html = "<li>No prediction scores available.</li>"
    score_plots_html = ""
    if not score_series.empty:
        score_mean = float(score_series.mean())
        score_p50 = float(score_series.quantile(0.5))
        score_p95 = float(score_series.quantile(0.95))
        score_min = float(score_series.min())
        score_max = float(score_series.max())
        score_metrics = [
            f"<li>Score mean: {score_mean:.4f}</li>",
            f"<li>Score p50: {score_p50:.4f}</li>",
            f"<li>Score p95: {score_p95:.4f}</li>",
            f"<li>Score min: {score_min:.4f}</li>",
            f"<li>Score max: {score_max:.4f}</li>",
        ]
        score_metrics_html = "\n".join(score_metrics)
        score_plot_path = plots_dir / "score_distribution.png"
        _plot_score_distribution(score_series, score_plot_path, bins=score_bins)
        score_plots_html = "<img src='plots/score_distribution.png' />"

    if not pred_series.empty:
        pred_rate = float(pred_series.mean())
        score_metrics_html += f"\n<li>Predicted default rate: {pred_rate:.2%}</li>"
        pred_plot_path = plots_dir / "prediction_rate.png"
        _plot_prediction_rate(pred_series, pred_plot_path)
        score_plots_html += "\n<img src='plots/prediction_rate.png' />"

    error_breakdown = _summarize_errors(meta_df_all[error_series >= 400])
    if error_breakdown:
        error_items = "\n".join(
            f"<li>{message} ({count})</li>" for message, count in error_breakdown
        )
        error_html = "<ul>\n" + error_items + "\n</ul>"
    else:
        error_html = "<p>No error details logged.</p>"

    drift_flags = summary_df.get("drift_detected", pd.Series(dtype=bool)).fillna(False)
    drift_count = int(drift_flags.sum())
    overall_drift = drift_count >= max(min_drift_features, 1) and n_prod >= min_prod_samples
    drift_features = summary_df.loc[drift_flags, "feature"].tolist() if not summary_df.empty else []

    dq_metrics = _summarize_data_quality(meta_df, production_df, sentinel_rates)
    if dq_metrics.get("source") == "none":
        dq_html = "<p>No data quality metrics available.</p>"
    else:
        dq_items = [
            f"<li>Source: {dq_metrics.get('source')}</li>",
            f"<li>Sample size: {dq_metrics.get('sample_size')}</li>",
            f"<li>Missing required rate: {dq_metrics.get('missing_required_rate', 0.0):.2%}</li>",
        ]
        if "invalid_numeric_rate" in dq_metrics:
            dq_items.append(f"<li>Invalid numeric rate: {dq_metrics.get('invalid_numeric_rate', 0.0):.2%}</li>")
        if "out_of_range_rate" in dq_metrics:
            dq_items.append(f"<li>Out-of-range rate: {dq_metrics.get('out_of_range_rate', 0.0):.2%}</li>")
        if "outlier_rate" in dq_metrics:
            dq_items.append(f"<li>Outlier rate: {dq_metrics.get('outlier_rate', 0.0):.2%}</li>")
        if "nan_rate" in dq_metrics:
            dq_items.append(f"<li>NaN rate (avg): {dq_metrics.get('nan_rate', 0.0):.2%}</li>")
        dq_items.append(
            f"<li>Unknown CODE_GENDER rate: {dq_metrics.get('unknown_gender_rate', 0.0):.2%}</li>"
        )
        dq_items.append(
            f"<li>Unknown FLAG_OWN_CAR rate: {dq_metrics.get('unknown_car_rate', 0.0):.2%}</li>"
        )
        dq_items.append(
            f"<li>DAYS_EMPLOYED sentinel rate: {dq_metrics.get('days_employed_sentinel_rate', 0.0):.2%}</li>"
        )
        dq_html = "<ul>\n" + "\n".join(dq_items) + "\n</ul>"

    summary_html = summary_df.to_html(index=False, escape=False)
    plots_html = "\n".join(
        f"<h4>{row['feature']}</h4><img src='plots/{_safe_name(row['feature'])}.png' />"
        for _, row in summary_df.iterrows()
    )

    sample_badge = ""
    if n_prod < min_prod_samples:
        sample_badge = (
            "<div class='badge warning'>Sample insuffisant: "
            f"{n_prod} &lt; {min_prod_samples} (resultats non fiables).</div>"
        )
    if n_prod < min_prod_samples:
        drift_badge = (
            "<div class='badge warning'>Drift non calcule "
            f"(n_prod &lt; {min_prod_samples}).</div>"
        )
    elif overall_drift:
        drift_badge = "<div class='badge alert'>Drift alert</div>"
    else:
        drift_badge = "<div class='badge ok'>No drift alert</div>"
    if not prod_since and not prod_until:
        window_info = "full_log"
    elif window_status in {"timestamp_missing", "timestamp_invalid"}:
        window_info = f"{window_status} (no filter applied)"
    else:
        window_info = f"{prod_since or '...'} to {prod_until or '...'}"

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Drift Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; }}
      th {{ background: #f3f3f3; }}
      img {{ max-width: 720px; }}
      .badge {{ display: inline-block; padding: 6px 10px; border-radius: 6px; font-weight: bold; margin: 6px 0; }}
      .badge.warning {{ background: #fde68a; color: #92400e; }}
      .badge.ok {{ background: #d1fae5; color: #065f46; }}
      .badge.alert {{ background: #fee2e2; color: #991b1b; }}
    </style>
  </head>
  <body>
    <h2>Production Monitoring Summary</h2>
    <ul>
      <li>Total calls (logged): {total_calls}</li>
      <li>Calls with inputs: {calls_with_inputs}</li>
      <li>Calls with data quality: {calls_with_dq}</li>
      <li>Calls success (status &lt; 400): {calls_success}</li>
      <li>Calls usable for drift: {n_prod}</li>
      <li>Error rate: {error_rate:.2%}</li>
      <li>Latency p50: {latency_p50:.2f} ms</li>
      <li>Latency p95: {latency_p95:.2f} ms</li>
    </ul>
    <h3>Top error reasons</h3>
    {error_html}
    {sample_badge}
    <h2>Score Monitoring</h2>
    <ul>
      {score_metrics_html}
    </ul>
    {score_plots_html}
    <h2>Data Quality</h2>
    {dq_html}
    <h2>Data Drift Summary</h2>
    {drift_badge}
    <ul>
      <li>Production sample size: {n_prod}</li>
      <li>Reference sample size: {n_ref}</li>
      <li>Reference dataset: {reference_path}</li>
      <li>Prod window: {window_info}</li>
      <li>Min prod sample: {min_prod_samples}</li>
      <li>PSI threshold: {psi_threshold}</li>
      <li>PSI epsilon: {psi_eps}</li>
      <li>Min category share: {min_category_share}</li>
      <li>FDR alpha: {fdr_alpha}</li>
      <li>Min drift features: {min_drift_features}</li>
      <li>Drifted features: {", ".join(drift_features) if drift_features else "None"}</li>
    </ul>
    {summary_html}
    <h2>Feature Distributions</h2>
    {plots_html}
  </body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a drift report from production logs.")
    parser.add_argument("--logs", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--reference", type=Path, default=Path("data/data_final.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--score-bins", type=int, default=30)
    parser.add_argument("--min-prod-samples", type=int, default=MIN_PROD_SAMPLES_DEFAULT)
    parser.add_argument("--psi-eps", type=float, default=PSI_EPS_DEFAULT)
    parser.add_argument("--min-category-share", type=float, default=RARE_CATEGORY_MIN_SHARE_DEFAULT)
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA_DEFAULT)
    parser.add_argument("--min-drift-features", type=int, default=1)
    parser.add_argument("--prod-since", type=str, default=None)
    parser.add_argument("--prod-until", type=str, default=None)
    args = parser.parse_args()

    report_path = generate_report(
        log_path=args.logs,
        reference_path=args.reference,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        psi_threshold=args.psi_threshold,
        score_bins=args.score_bins,
        min_prod_samples=args.min_prod_samples,
        psi_eps=args.psi_eps,
        min_category_share=args.min_category_share,
        fdr_alpha=args.fdr_alpha,
        min_drift_features=args.min_drift_features,
        prod_since=args.prod_since,
        prod_until=args.prod_until,
    )
    print(f"Drift report saved to {report_path}")


if __name__ == "__main__":
    main()
