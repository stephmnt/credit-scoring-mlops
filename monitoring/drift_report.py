# scripts/drift_report.py

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# On peut réutiliser les fonctions de normalisation si elles sont dans un module partagé
# Pour cet exemple, je les recopie ici pour la clarté.

def _normalize_gender(series: pd.Series) -> pd.Series:
    """Normalise la colonne CODE_GENDER."""
    return series.str.upper().replace({'XNA': 'F', 'MALE': 'M', 'FEMALE': 'F'})

def _replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace la valeur sentinelle dans DAYS_EMPLOYED."""
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: pd.NA})
    return df

def _load_logs(log_path: Path) -> pd.DataFrame | None:
    """Charge les inputs depuis les logs JSONL."""
    if not log_path.exists():
        print(f"Warning: Log file not found at {log_path}")
        return None
    
    records = []
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                # On ne garde que les appels réussis avec des inputs
                if log_entry.get('status_code', 500) < 400 and 'inputs' in log_entry:
                    records.append(log_entry['inputs'])
            except json.JSONDecodeError:
                continue # Ignore les lignes mal formées
    
    if not records:
        return None
        
    return pd.DataFrame(records)


def generate_evidently_report(
    reference_path: Path,
    log_path: Path,
    output_path: Path,
    sample_size: int = 20000,
):
    """Génère un rapport de dérive de données avec Evidently."""
    
    print("1. Chargement des données de référence...")
    reference_df = pd.read_parquet(reference_path)
    if sample_size > 0 and sample_size < len(reference_df):
        reference_df = reference_df.sample(n=sample_size, random_state=42)

    print("2. Chargement et préparation des données de production (logs)...")
    prod_df = _load_logs(log_path)
    
    if prod_df is None or prod_df.empty:
        print("Aucune donnée de production valide trouvée. Le rapport ne peut être généré.")
        return

    # Appliquer le même pré-traitement pour une comparaison juste
    reference_df = _replace_sentinel(reference_df)
    prod_df = _replace_sentinel(prod_df)
    
    if 'CODE_GENDER' in reference_df.columns:
        reference_df['CODE_GENDER'] = _normalize_gender(reference_df['CODE_GENDER'])
    if 'CODE_GENDER' in prod_df.columns:
        prod_df['CODE_GENDER'] = _normalize_gender(prod_df['CODE_GENDER'])

    # S'assurer que les colonnes correspondent
    # Evidently gère les colonnes manquantes, mais c'est une bonne pratique
    ref_cols = set(reference_df.columns)
    prod_cols = set(prod_df.columns)
    common_cols = list(ref_cols.intersection(prod_cols))
    
    reference_df = reference_df[common_cols]
    prod_df = prod_df[common_cols]

    print(f"3. Génération du rapport de dérive sur {len(common_cols)} features communes...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(reference_data=reference_df, current_data=prod_df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_drift_report.save_html(str(output_path))
    
    print(f"✅ Rapport de dérive généré avec succès : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a data drift report using Evidently AI.")
    parser.add_argument("--reference-path", type=Path, default="data/data_final.parquet", help="Path to the reference dataset.")
    parser.add_argument("--log-path", type=Path, default="logs/predictions.jsonl", help="Path to the production log file.")
    parser.add_argument("--output-path", type=Path, default="reports/evidently_drift_report.html", help="Path to save the HTML report.")
    parser.add_argument("--sample-size", type=int, default=20000, help="Size of the reference sample to use (0 for full dataset).")
    
    args = parser.parse_args()

    generate_evidently_report(
        reference_path=args.reference_path,
        log_path=args.log_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
    )
