# Profiling & Optimisation (Etape 4)

## Objectif

Mesurer la latence d'inference, identifier les goulots d'etranglement et proposer une optimisation logicielle sans regression fonctionnelle.

## Setup

- Script (archivé): `dev_archive/profiling/profile_inference.py`
- Workflow courant: notebook modélisation (section TODO 5)
- Donnees: `data/data_final.parquet` (echantillon)
- Parametres: `--sample-size 500 --batch-size 100 --runs 2`
- Modele: `HistGB_final_model.pkl`

Les resultats sont sauvegardes dans:

- `docs/performance/benchmark_results.json`
- `docs/performance/profile_summary.txt`

## Resultats

| Scenario | Batch | Mean (ms) | P50 (ms) | P95 (ms) | Throughput (rows/s) |
| --- | --- | ---:| ---:| ---:| ---:|
| optimized_preprocess | 100 | 187.37 | 169.96 | 271.41 | 533.71 |
| legacy_preprocess_alignment | 100 | 273.05 | 264.45 | 357.41 | 366.23 |

Gain observe (moyenne): ~31% de reduction de latence par batch sur le chemin optimise.

## Goulots d'etranglement (cProfile)

Extrait `docs/performance/profile_summary.txt`:

- `app.main:preprocess_input` represente l'essentiel du temps cumule (~0.90s sur 1.05s).
- Operations pandas dominantes:
  - `DataFrame.__setitem__` / `insert`
  - `fillna`, `to_numeric`
  - `get_dummies`
- `HistGradientBoostingClassifier.predict_proba` est present mais non majoritaire (~0.15s).

## Optimisation appliquee

- Alignement one-hot optimise: remplacement de la boucle d'ajout de colonnes par un `reindex` avec `fill_value=0`.
- Alignement des colonnes d'entree: remplacement de l'ajout colonne-par-colonne par un `reindex` sur `columns_keep`.
- Resultat: latence moyenne par batch reduite vs le chemin legacy (mesure ci-dessus).

## Pistes futures

- Precalculer un pipeline scikit-learn complet (OneHotEncoder + scaler) pour eviter le `get_dummies` a chaque requete.
- Export ONNX et inference via ONNX Runtime pour accelerer la predicition.
- Ajuster la taille de batch pour maximiser le throughput.
- Eventuellement degrader certains controles en mode "fast" si le contexte le permet (trade-off securite vs latence).
