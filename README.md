---
title: OCR Projet 06
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OCR Projet 06 – Crédit

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet06/deploy.yml)](https://github.com/stephmnt/OCR_Projet05/actions/workflows/deploy.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/OCR_Projet06?display_date=published_at&style=flat-square)](https://github.com/stephmnt/OCR_Projet06/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/OCR_projet06.svg)](https://github.com/stephmnt/OCR_Projet06/blob/main/LICENSE)

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

L'API attend un payload JSON avec une cle `data`. La valeur peut etre un objet
unique (un client) ou une liste d'objets (plusieurs clients). La liste des
features requises (jeu reduit) est disponible via l'endpoint `/features`. Les
autres champs sont optionnels et seront completes par des valeurs par defaut.

Inputs minimums (10 + `SK_ID_CURR`) :

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

### Environnement Poetry (recommande)

Le fichier `pyproject.toml` fixe des versions compatibles pour un stack recent
(`numpy>=2`, `pyarrow>=15`, `scikit-learn>=1.6`). L'environnement vise Python
3.11.

```shell
poetry env use 3.11
poetry install
poetry run pytest -q
poetry run uvicorn app.main:app --reload --port 7860
```

Important : le modele `HistGB_final_model.pkl` doit etre regenere avec la
nouvelle version de scikit-learn (re-execution de
`P6_MANET_Stephane_notebook_modélisation.ipynb`, cellule de sauvegarde pickle).

Note : `requirements.txt` est aligne sur `pyproject.toml` (meme versions).

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

Note : l'API valide strictement les champs requis (`/features`). Pour afficher
toutes les colonnes possibles : `/features?include_all=true`.

### Demo live (commandes cles en main)

Lancer l'API :

```shell
uvicorn app.main:app --reload --port 7860
```

Verifier le service :

```shell
curl -s http://127.0.0.1:7860/health
```

Voir les features attendues :

```shell
curl -s http://127.0.0.1:7860/features
```

Predire un client :

```shell
curl -s -X POST "http://127.0.0.1:7860/predict?threshold=0.5" \
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

## Contenu de la release

- **Preparation + pipeline** : nettoyage / preparation, encodage, imputation et pipeline d'entrainement presentes.
- **Gestion du desequilibre** : un sous-echantillonnage est applique sur le jeu d'entrainement final.
- **Comparaison multi-modeles** : baseline, Naive Bayes, Logistic Regression, Decision Tree, Random Forest,
  HistGradientBoosting, LGBM, XGB sont compares.
- **Validation croisee + tuning** : `StratifiedKFold`, `GridSearchCV` et Hyperopt sont utilises.
- **Score metier + seuil optimal** : le `custom_score` est la metrique principale des tableaux de comparaison et de la CV, avec un `best_threshold` calcule.
- **Explicabilite** : feature importance, SHAP et LIME sont inclus.
- **MLOps (MLflow)** : tracking des params / metriques (dont `custom_score` et `best_threshold`), tags,
  registry et passage en "Staging".

![Screenshot MLFlow](https://raw.githubusercontent.com/stephmnt/OCR_Projet06/main/screen-mlflow.png)

## Réduction des features

Réduction des features : l’API utilise un top‑10 SHAP, alors que la mission insiste sur une réduction à l’aide d’une matrice de corrélation. La corrélation est bien documentée dans le notebook d’exploration, mais la liste utilisée par l’API n’est pas explicitement issue de cette matrice. À clarifier dans la doc ou aligner la sélection sur la corrélation.

## Glossaire rapide

- **custom_score** : metrique metier qui penalise plus fortement les faux negatifs que les faux positifs.
- **Seuil optimal** : probabilite qui sert a transformer un score en classe 0/1.
- **Validation croisee (CV)** : evaluation sur plusieurs sous-echantillons pour eviter un resultat "chanceux".
- **MLflow tracking** : historique des runs, parametres et metriques.
- **Registry** : espace MLflow pour versionner et promouvoir un modele (ex. "Staging").
