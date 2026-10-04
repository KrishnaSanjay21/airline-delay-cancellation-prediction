# Airline Delay and Cancellation Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)](https://streamlit.io/)
[![Data: BTS + NOAA](https://img.shields.io/badge/data-BTS%20%2B%20NOAA-0F766E)](https://www.transtats.bts.gov/ontime/Index.aspx)

A reproducible data-science project combining public BTS on-time performance records with NOAA airport weather to predict arrival delays of at least 15 minutes and flight cancellations.

> Portfolio demonstration only. Predictions are not operational dispatch guidance.

## What this project demonstrates

- Real-data ingestion from BTS monthly on-time ZIP archives and the NOAA Climate Data Online API
- Origin- and destination-airport weather joins for 25 major U.S. airports
- Explicit feature availability and outcome-leakage controls
- Whole-date chronological train, probability-calibration, and untouched test periods
- Class-weighted logistic regression, random forest, and gradient boosting
- Held-out sigmoid calibration for every candidate
- ROC AUC, average precision, Brier score, precision, recall, F1, and confusion counts
- Performance and prevalence monitoring by origin airport
- Model-agnostic permutation importance
- A filterable Streamlit operations and model-validation dashboard
- SQL data-quality queries, Docker packaging, tests, and refresh templates

## Architecture

```text
BTS monthly ZIPs ─┐
                  ├─> validated Parquet ─> as-of feature table
NOAA CDO API ─────┘                            │
                                              ├─ chronological train/calibration/test
                                              ├─ logistic / random forest / gradient boost
                                              └─ metrics + Streamlit operations dashboard
```

See [docs/architecture.md](docs/architecture.md) for the prediction point, validation design, and production upgrade path.

## Quick start

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. Configure

Request a free NOAA token, copy `.env.example` to `.env`, and select a BTS period that has already been published.

```powershell
Copy-Item .env.example .env
```

```dotenv
NOAA_TOKEN="your-private-token"
START_DATE="2024-01-01"
END_DATE="2024-03-31"
TOP_AIRPORTS=25
```

### 3. Run end to end

```powershell
airline-risk run-all
streamlit run app.py
```

The pipeline writes raw API/download caches under `data/raw`, processed Parquet tables under `data/processed`, and evaluated models/metrics under `artifacts`. These generated files are excluded from Git.

Individual stages can be rerun safely:

```powershell
airline-risk ingest
airline-risk build-features
airline-risk train
```

## Modeling contract

| Component | Implementation |
|---|---|
| Delay outcome | BTS `ArrDel15 = 1`; cancelled flights excluded |
| Cancellation outcome | BTS `Cancelled = 1` |
| Prediction point | Scheduled departure |
| Train/calibration/test | Earliest 70% / next 15% / latest 15% of whole dates |
| Candidate models | Logistic regression, random forest, histogram gradient boosting |
| Probability calibration | Sigmoid scaling on the chronological calibration period |
| Primary ranking metric | Average precision |
| Additional evaluation | ROC AUC, Brier, precision, recall, F1, calibration curve, airport metrics |
| Explainability | Permutation importance on the chronological test period |

Actual delay, cancellation-reason, diverted-status, and realized-time fields are explicitly prohibited from model inputs. See the [data dictionary](docs/data-dictionary.md) and [model card](docs/model-card.md).

## Dashboard

The Streamlit dashboard includes:

- Operations overview with observed and predicted disruption trends
- Model comparison and probability calibration
- Permutation feature importance
- Airport-level discrimination and prevalence
- Threshold-based inspection of scored holdout flights

It displays historical holdout predictions, not live flight forecasts. A deployed operational system would need forecast weather available at scoring time rather than completed-day observations.

## Testing

```powershell
ruff check src tests app.py
pytest
```

The suite validates BTS archive parsing, NOAA authentication/pagination behavior, weather normalization, two-airport joins, target rules, chronological boundaries, leakage controls, and evaluation metrics without downloading live data.

## Repository map

```text
app.py                    Streamlit dashboard
config/                   airport-to-NOAA-station mapping
docs/                     architecture, dictionary, model card, runbook
orchestration/            activation-ready GitHub Actions templates
sql/                      DuckDB data-quality checks
src/airline_risk/         ingestion, features, modeling, evaluation, CLI
tests/                    deterministic test suite
Dockerfile                dashboard container
docker-compose.yml        pipeline and dashboard services
```

## Data sources

- [BTS On-Time Performance data](https://www.transtats.bts.gov/ontime/Index.aspx)
- [BTS monthly PREZIP archives](https://transtats.bts.gov/PREZIP/)
- [NOAA Climate Data Online API v2](https://www.ncei.noaa.gov/cdo-web/webservices/v2)

NOAA CDO requires a token and publishes request limits. Secrets and downloaded data are intentionally excluded from this repository. Review the source agencies' current documentation and terms before large-scale use.

## Deployment status

The code, dashboard, Docker configuration, tests, and automation templates are complete. No live dashboard URL, recurring refresh, or production prediction service is claimed until real-data artifacts, secrets, hosting, and monitoring are configured and verified.

The implementation is developed in milestone commits so the design history is reviewable.

