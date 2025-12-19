# OCR Projet 06 – Crédit

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/OCR_Projet06/deploy.yml)](https://github.com/stephmnt/OCR_Projet05/actions/workflows/deploy.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/OCR_Projet06?display_date=published_at&style=flat-square)](https://github.com/stephmnt/OCR_Projet06/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/OCR_projet06.svg)](https://github.com/stephmnt/OCR_Projet06/blob/main/LICENSE)

## Lancer MLFlow

```shell
mlflow server
```

```shell
mlflow models serve -m "models:/credit_scoring_model/Staging" -p 5001 --no-conda
```

```shell
mlflow ui --backend-store-uri "file:${PWD}/mlruns" --port 5000
```
