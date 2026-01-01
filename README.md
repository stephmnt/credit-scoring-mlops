---
title: Credit scoring MLOps
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Credit scoring MLOps

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/credit-scoring-mlops/deploy.yml)](https://github.com/stephmnt/credit-scoring-mlops/actions/workflows/deploy.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/credit-scoring-mlops?display_date=published_at&style=flat-square)](https://github.com/stephmnt/credit-scoring-mlops/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/credit-scoring-mlops.svg)](https://github.com/stephmnt/credit-scoring-mlops/blob/main/LICENSE)

## Lancer MLFlow

Le notebook est configure pour utiliser un serveur MLflow local (`http://127.0.0.1:5000`).
Pour voir les runs et creer l'experiment, demarrer le serveur avec le meme backend :

```shell
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "file:${PWD}/mlruns" \
  --default-artifact-root "file:${PWD}/mlruns"
```

Seulement l'interface (sans API), lancer :

```shell
mlflow ui --backend-store-uri "file:${PWD}/mlruns" --port 5000
```

Pour tester le serving du modele en staging :

```shell
mlflow models serve -m "models:/credit_scoring_model/Staging" -p 5001 --no-conda
```

## API FastAPI

L'API attend un payload JSON avec une cle `data`. La valeur peut etre un objet unique (un client) ou une liste d'objets (plusieurs clients). La liste des features requises (jeu reduit) est disponible via l'endpoint `/features`. Les autres champs sont optionnels et seront completes par des valeurs par defaut.

Inputs minimums (10 + `SK_ID_CURR`) derives d'une selection par correlation (voir `/features`) :

- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `AMT_ANNUITY`
- `EXT_SOURCE_1`
- `CODE_GENDER`
- `DAYS_EMPLOYED`
- `AMT_CREDIT`
- `AMT_GOODS_PRICE`
- `DAYS_BIRTH`
- `FLAG_OWN_CAR`

Parametres utiles (selection des features) :

- `FEATURE_SELECTION_METHOD` (defaut: `correlation`)
- `FEATURE_SELECTION_TOP_N` (defaut: `8`)
- `FEATURE_SELECTION_MIN_CORR` (defaut: `0.02`)

### Environnement pip (dev)

Le developpement local utilise pip et `requirements.txt` (versions figees),
avec Python 3.11+.

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
uvicorn app.main:app --reload --port 7860
```

### Environnement Poetry (livrable)

Le livrable inclut `pyproject.toml`, aligne sur `requirements.txt`. Si besoin :

```shell
poetry install --with dev
poetry run pytest -q
poetry run uvicorn app.main:app --reload --port 7860
```

Important : le modele `HistGB_final_model.pkl` doit etre regenere avec la
version de scikit-learn definie dans `requirements.txt` / `pyproject.toml`
(re-execution de `P6_MANET_Stephane_notebook_modélisation.ipynb`, cellule de
sauvegarde pickle).

### Exemple d'input (schema + valeurs)

Schema :

```json
{
  "data": {
    "SK_ID_CURR": "int",
    "EXT_SOURCE_2": "float",
    "EXT_SOURCE_3": "float",
    "AMT_ANNUITY": "float",
    "EXT_SOURCE_1": "float",
    "CODE_GENDER": "str",
    "DAYS_EMPLOYED": "int",
    "AMT_CREDIT": "float",
    "AMT_GOODS_PRICE": "float",
    "DAYS_BIRTH": "int",
    "FLAG_OWN_CAR": "str"
  }
}
```

Valeurs d'exemple :

```json
{
  "data": {
    "SK_ID_CURR": 100002,
    "EXT_SOURCE_2": 0.61,
    "EXT_SOURCE_3": 0.75,
    "AMT_ANNUITY": 24700.5,
    "EXT_SOURCE_1": 0.45,
    "CODE_GENDER": "M",
    "DAYS_EMPLOYED": -637,
    "AMT_CREDIT": 406597.5,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "FLAG_OWN_CAR": "N"
  }
}
```

### Prediction minimale (client existant)

Endpoint `POST /predict-minimal` : l'utilisateur fournit un identifiant client,
un montant de credit et une duree. Les autres features sont prises depuis la
reference clients (`CUSTOMER_DATA_PATH`, par defaut `data/data_final.parquet`).
Si la reference est absente, l'API renvoie 503.

```shell
curl -s -X POST "${BASE_URL}/predict-minimal" \
  -H "Content-Type: application/json" \
  -d '{
    "sk_id_curr": 100001,
    "amt_credit": 200000,
    "duration_months": 60
  }'
```

Variables utiles :

- `CUSTOMER_LOOKUP_ENABLED=1` active la recherche client (defaut: 1)
- `CUSTOMER_DATA_PATH=data/data_final.parquet`
- `CUSTOMER_LOOKUP_CACHE=1` garde la reference en memoire

### Data contract (validation)

- Types numeriques stricts (invalides -> 422).
- Ranges numeriques (min/max entrainement) controles.
- Categoriels normalises: `CODE_GENDER` -> {`F`, `M`}, `FLAG_OWN_CAR` -> {`Y`, `N`}.
- Sentinelle `DAYS_EMPLOYED=365243` remplacee par NaN.
- Logs enrichis via `data_quality` pour distinguer drift vs qualite de donnees.

### Interface Gradio (scoring)

```shell
python gradio_app.py
```

Sur Hugging Face Spaces, `app.py` lance l'UI Gradio automatiquement.

Note : l'API valide strictement les champs requis (`/features`). Pour afficher
toutes les colonnes possibles : `/features?include_all=true`.

### Hugging Face (assets lourds)

Les fichiers binaires (modele, preprocessor, data_final) ne sont pas pushes
dans le Space. Ils sont telecharges a l'execution via Hugging Face Hub si les
variables suivantes sont definies :

- `HF_MODEL_REPO_ID` + `HF_MODEL_FILENAME` + `HF_MODEL_REPO_TYPE`
- `HF_PREPROCESSOR_REPO_ID` + `HF_PREPROCESSOR_FILENAME` + `HF_PREPROCESSOR_REPO_TYPE`
- `HF_CUSTOMER_REPO_ID` + `HF_CUSTOMER_FILENAME` + `HF_CUSTOMER_REPO_TYPE`

Exemple (un seul repo dataset avec 3 fichiers) :

- `HF_MODEL_REPO_ID=stephmnt/credit-scoring-mlops-assets`
- `HF_MODEL_REPO_TYPE=dataset`
- `HF_MODEL_FILENAME=HistGB_final_model.pkl`
- `HF_PREPROCESSOR_REPO_ID=stephmnt/credit-scoring-mlops-assets`
- `HF_PREPROCESSOR_REPO_TYPE=dataset`
- `HF_PREPROCESSOR_FILENAME=preprocessor.joblib`
- `HF_CUSTOMER_REPO_ID=stephmnt/credit-scoring-mlops-assets`
- `HF_CUSTOMER_REPO_TYPE=dataset`
- `HF_CUSTOMER_FILENAME=data_final.parquet`

### Demo live (commandes cles en main)

Lancer l'API :

```shell
uvicorn app.main:app --reload --port 7860
```

Verifier le service (HF) :

```shell
BASE_URL="https://stephmnt-credit-scoring-mlops.hf.space"
API_BASE="${BASE_URL}/api"
curl -s "${API_BASE}/health"
```

Note : sur HF Spaces, l'UI Gradio est a la racine, l'API est sous `/api`.

Voir les features attendues (HF) :

```shell
curl -s "${API_BASE}/features"
```

Predire un client (HF) :

```shell
curl -s -X POST "${API_BASE}/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "SK_ID_CURR": 100002,
      "EXT_SOURCE_2": 0.61,
      "EXT_SOURCE_3": 0.75,
      "AMT_ANNUITY": 24700.5,
      "EXT_SOURCE_1": 0.45,
      "CODE_GENDER": "M",
      "DAYS_EMPLOYED": -637,
      "AMT_CREDIT": 406597.5,
      "AMT_GOODS_PRICE": 351000.0,
      "DAYS_BIRTH": -9461,
      "FLAG_OWN_CAR": "N"
    }
  }'
```

Predire plusieurs clients (batch, HF) :

```shell
curl -s -X POST "${API_BASE}/predict?threshold=0.45" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "SK_ID_CURR": 100002,
        "EXT_SOURCE_2": 0.61,
        "EXT_SOURCE_3": 0.75,
        "AMT_ANNUITY": 24700.5,
        "EXT_SOURCE_1": 0.45,
        "CODE_GENDER": "M",
        "DAYS_EMPLOYED": -637,
        "AMT_CREDIT": 406597.5,
        "AMT_GOODS_PRICE": 351000.0,
        "DAYS_BIRTH": -9461,
        "FLAG_OWN_CAR": "N"
      },
      {
        "SK_ID_CURR": 100003,
        "EXT_SOURCE_2": 0.52,
        "EXT_SOURCE_3": 0.64,
        "AMT_ANNUITY": 19000.0,
        "EXT_SOURCE_1": 0.33,
        "CODE_GENDER": "F",
        "DAYS_EMPLOYED": -1200,
        "AMT_CREDIT": 320000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "DAYS_BIRTH": -12000,
        "FLAG_OWN_CAR": "Y"
      }
    ]
  }'
```

Exemple d'erreur (champ requis manquant, HF) :

```shell
curl -s -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "EXT_SOURCE_2": 0.61
    }
  }'
```

## Monitoring & Data Drift (Etape 3)

L'API enregistre les appels `/predict` en JSONL (inputs, outputs, latence).
Par defaut, les logs sont stockes dans `logs/predictions.jsonl`.

Variables utiles :

- `LOG_PREDICTIONS=1` active l'ecriture des logs (defaut: 1)
- `LOG_DIR=logs`
- `LOG_FILE=predictions.jsonl`
- `LOGS_ACCESS_TOKEN` pour proteger l'endpoint `/logs`
- `LOG_HASH_SK_ID=1` pour anonymiser `SK_ID_CURR`

Les logs incluent un bloc `data_quality` par requete (champs manquants,
types invalides, out-of-range, categories inconnues, sentinelle
`DAYS_EMPLOYED`).

Exemple local :

```shell
LOG_PREDICTIONS=1 LOG_DIR=logs uvicorn app.main:app --reload --port 7860
```

Recuperer les logs (HF) :

Configurer `LOGS_ACCESS_TOKEN` dans les secrets du Space, puis :

```shell
curl -s -H "X-Logs-Token: $LOGS_ACCESS_TOKEN" "${API_BASE}/logs?tail=200"
```

Alternative :

```shell
curl -s -H "Authorization: Bearer $LOGS_ACCESS_TOKEN" "${API_BASE}/logs?tail=200"
```

Apres quelques requêtes, générer le rapport de drift :

```shell
python monitoring/drift_report.py \
  --logs logs/predictions.jsonl \
  --reference data/data_final.parquet \
  --output-dir reports \
  --min-prod-samples 200 \
  --fdr-alpha 0.05 \
  --prod-since "2024-01-01T00:00:00Z" \
  --prod-until "2024-01-31T23:59:59Z"
```

Le rapport HTML est généré dans `reports/drift_report.html` (avec des plots dans
`reports/plots/`). Sur Hugging Face, le disque est éphemère : télécharger les logs
avant d'analyser.

Le drift est calcule uniquement si `n_prod >= --min-prod-samples` (defaut 200).
Sinon, un badge "Sample insuffisant" est affiche et les alertes sont desactivees.

Robustesse integree:

- Categoriels: PSI avec lissage (`--psi-eps`) + categories rares regroupees (OTHER).
- Numeriques: KS corrige par FDR (Benjamini-Hochberg, `--fdr-alpha`).
- Sentinel `DAYS_EMPLOYED`: converti en NaN + taux suivi.

Le rapport inclut aussi la distribution des scores predits et le taux de prediction
(option `--score-bins` pour ajuster le nombre de bins), ainsi qu'une section
Data Quality si les logs contiennent `data_quality` (types, NaN, out-of-range,
categories inconnues).

Pour simuler des fenetres glissantes, utiliser `--prod-since` / `--prod-until`
avec les timestamps des logs.

Runbook drift: `docs/monitoring/runbook.md`.

Captures (snapshot local du reporting + stockage):

- Rapport: `docs/monitoring/drift_report.html` + `docs/monitoring/plots/`
- Stockage des logs: `docs/monitoring/logs_storage.png`

## Profiling & Optimisation (Etape 4)

Profiling et benchmark d'inference (cProfile + latence) :

```shell
python profiling/profile_inference.py \
  --sample-size 2000 \
  --batch-size 128 \
  --runs 3
```

Sorties:

- `docs/performance/benchmark_results.json`
- `docs/performance/profile_summary.txt`
- Rapport detaille: `docs/performance/performance_report.md`

Dashboard local Streamlit (monitoring + drift):

```shell
python -m streamlit run monitoring/streamlit_app.py
```

## Contenu de la release

- **Preparation + pipeline** : nettoyage / preparation, encodage, imputation et pipeline d'entrainement presentes.
- **Gestion du desequilibre** : un sous-echantillonnage est applique sur le jeu d'entrainement final.
- **Comparaison multi-modeles** : baseline, Naive Bayes, Logistic Regression, Decision Tree, Random Forest,
  HistGradientBoosting, LGBM, XGB sont compares.
- **Validation croisee + tuning** : `StratifiedKFold`, `GridSearchCV` et Hyperopt sont utilises.
- **Score metier + seuil optimal** : le `custom_score` est la metrique principale des tableaux de comparaison et de la CV, avec un `best_threshold` calcule.
- **Explicabilite** : feature importance, SHAP et LIME sont inclus.
- **Selection de features par correlation** : top‑N numeriques + un petit set categoriel, expose via `/features`.
- **Interface Gradio** : formulaire minimal (id client + montant + duree) base sur la reference clients.
- **Monitoring & drift** : rapport HTML avec gating par volume, PSI robuste, KS + FDR, data quality et
  distribution des scores (snapshots dans `docs/monitoring/`).
- **Profiling & optimisation** : benchmark d'inference + profil cProfile (dossier `docs/performance/`).
- **CI/CD** : tests avec couverture (`pytest-cov`), build Docker et deploy vers Hugging Face Spaces.

![Screenshot MLFlow](https://raw.githubusercontent.com/stephmnt/credit-scoring-mlops/main/screen-mlflow.png)

### Manques prioritaires

* Mission 2 Étape 4 non couverte: pas de profiling/optimisation post‑déploiement ni rapport de gains, à livrer avec une version optimisée.

### Preuves / doc à compléter

* Lien explicite vers le dépôt public + stratégie de versions/branches à ajouter dans README.md.
* Preuve de model registry/serving MLflow à conserver (capture UI registry ou commande de serving) en plus de screen-mlflow.png.
* Dataset de référence non versionné (data_final.parquet est ignoré), documenter l’obtention pour exécuter drift_report.py.
* Badge GitHub Actions pointe vers OCR_Projet05 dans README.md, corriger l’URL.
* RGPD/PII: LOG_HASH_SK_ID est désactivé par défaut dans main.py, préciser l’activation en prod dans README.md.

### Améliorations recommandées

* Compléter les tests API: /logs (auth OK/KO), batch predict, param threshold, SK_ID_CURR manquant, outliers dans test_api.py.
* Simplifier le fallback ALLOW_MISSING_ARTIFACTS et DummyModel si les artefacts sont versionnés (nettoyer main.py et conftest.py).
* Si l’évaluateur attend une stratégie de branches, créer une branche feature et fusionner pour preuve.
