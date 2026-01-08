# Profiling & Optimisation (Etape 4)

## Objectif

Mesurer la latence d'inference, identifier les goulots d'etranglement et proposer une optimisation logicielle sans regression fonctionnelle.

## Setup

- Notebook: `notebooks/P6_MANET_Stephane_notebook_modélisation.ipynb` (section TODO 5)
- Donnees: `data/data_final.parquet` (echantillon)
- Parametres: `--sample-size 500 --batch-size 100 --runs 2`
- Modele: `data/*_final_model.pkl` (ex: `data/xgb_final_model.pkl`)

Les resultats sont sauvegardes dans:

- `docs/performance/benchmark_results.json`
- `docs/performance/profile_summary.txt`

## Resultats

| Scenario | Batch | Mean (ms) | P50 (ms) | P95 (ms) | Throughput (rows/s) |
| --- | --- | ---:| ---:| ---:| ---:|
| optimized_preprocess | 100 | 35.73 | 33.77 | 43.09 | 2798.44 |
| legacy_preprocess_alignment | 100 | 47.57 | 47.19 | 51.23 | 2102.36 |

Gain observe (moyenne): ~25% de reduction de latence par batch sur le chemin optimise.

## Goulots d'etranglement (cProfile)

Extrait `docs/performance/profile_summary.txt`:

- `app.main:preprocess_input` represente l'essentiel du temps cumule (voir `docs/performance/profile_summary.txt`).
- Operations pandas dominantes:
  - `DataFrame.__setitem__` / `insert`
  - `fillna`, `to_numeric`
  - `get_dummies`
- `predict_proba` est present mais non majoritaire.

## Optimisation appliquee

- Alignement one-hot optimise: remplacement de la boucle d'ajout de colonnes par un `reindex` avec `fill_value=0`.
- Alignement des colonnes d'entree: remplacement de l'ajout colonne-par-colonne par un `reindex` sur `columns_keep`.
- Resultat: latence moyenne par batch reduite vs le chemin legacy (mesure ci-dessus).

## Pistes futures

- Precalculer un pipeline scikit-learn complet (OneHotEncoder + scaler) pour eviter le `get_dummies` a chaque requete.
- Export ONNX et inference via ONNX Runtime pour accelerer la predicition.
- Ajuster la taille de batch pour maximiser le throughput.
- Eventuellement degrader certains controles en mode "fast" si le contexte le permet (trade-off securite vs latence).
