import json
from pathlib import Path

import pandas as pd

from monitoring.drift_report import generate_report


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def test_generate_report_insufficient_sample(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    entries = [
        {
            "inputs": {
                "EXT_SOURCE_2": 0.45,
                "EXT_SOURCE_3": 0.62,
                "AMT_ANNUITY": 24700.5,
                "EXT_SOURCE_1": 0.41,
                "CODE_GENDER": "F",
                "DAYS_EMPLOYED": -1200,
                "AMT_CREDIT": 406597.5,
                "AMT_GOODS_PRICE": 351000.0,
                "DAYS_BIRTH": -9461,
                "FLAG_OWN_CAR": "N",
            },
            "status_code": 200,
            "probability": 0.4,
            "prediction": 0,
        }
    ]
    _write_jsonl(log_path, entries)

    reference_df = pd.DataFrame(
        [
            {
                "EXT_SOURCE_2": 0.44,
                "EXT_SOURCE_3": 0.61,
                "AMT_ANNUITY": 25500.0,
                "EXT_SOURCE_1": 0.39,
                "CODE_GENDER": "F",
                "DAYS_EMPLOYED": -1500,
                "AMT_CREDIT": 405000.0,
                "AMT_GOODS_PRICE": 350000.0,
                "DAYS_BIRTH": -9500,
                "FLAG_OWN_CAR": "N",
            },
            {
                "EXT_SOURCE_2": 0.33,
                "EXT_SOURCE_3": 0.55,
                "AMT_ANNUITY": 21000.0,
                "EXT_SOURCE_1": 0.35,
                "CODE_GENDER": "M",
                "DAYS_EMPLOYED": -2000,
                "AMT_CREDIT": 300000.0,
                "AMT_GOODS_PRICE": 250000.0,
                "DAYS_BIRTH": -10000,
                "FLAG_OWN_CAR": "Y",
            },
        ]
    )
    reference_path = tmp_path / "reference.parquet"
    reference_df.to_parquet(reference_path)

    output_dir = tmp_path / "reports"
    report_path = generate_report(
        log_path=log_path,
        reference_path=reference_path,
        output_dir=output_dir,
        sample_size=10,
        psi_threshold=0.2,
        score_bins=10,
        min_prod_samples=200,
    )
    html = report_path.read_text(encoding="utf-8")
    assert "Sample insuffisant" in html
