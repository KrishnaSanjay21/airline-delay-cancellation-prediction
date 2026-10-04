# Model card

## Purpose

Estimate pre-departure risk of (1) arrival delay of at least 15 minutes and (2) cancellation. The project demonstrates reproducible classification, temporal validation, calibration, and operations-oriented monitoring.

## Candidate models

- Class-weighted logistic regression with one-hot categoricals
- Class-balanced random forest
- Class-balanced histogram gradient boosting

Each base model is trained on the earliest period. Sigmoid calibration is learned on the next period. Model selection uses average precision on the untouched test period, with ROC AUC and Brier score reported alongside it.

## Intended use

- Portfolio demonstration and retrospective operational analysis
- Comparing relative disruption patterns by carrier, route, schedule, weather, and airport
- Demonstrating how to monitor discrimination and calibration across airports

## Not intended for

- Live dispatch, passenger rebooking, safety decisions, or regulatory reporting
- Causal claims about carriers or airports
- Use outside the represented routes, periods, or weather regimes without drift testing

## Limitations

- NOAA historical daily observations are not weather forecasts. A production pre-departure service must use forecast snapshots available at scoring time.
- BTS publication lag means the pipeline is retrospective rather than real-time.
- Class weighting changes optimization behavior; calibrated probabilities are evaluated separately.
- Airport metrics can be unstable for small samples, so the dashboard suppresses groups below a minimum count.
- Extreme disruptions, policy changes, route changes, and new carriers can create temporal drift.

## Monitoring

Track input missingness, target prevalence, probability distributions, Brier score, average precision, ROC AUC, calibration by decile, and airport-level performance. Retrain only after investigating material drift and preserving a newer chronological holdout.

