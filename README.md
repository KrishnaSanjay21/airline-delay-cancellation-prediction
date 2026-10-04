# Airline Delay and Cancellation Prediction

A reproducible data-science project combining public BTS on-time performance records with NOAA airport weather to predict arrival delays of at least 15 minutes and flight cancellations.

> Portfolio demonstration only. Predictions are not operational dispatch guidance.

## Architecture

```text
BTS monthly ZIPs ─┐
                  ├─> validated Parquet ─> as-of feature table
NOAA CDO API ─────┘                            │
                                              ├─ chronological train/calibration/test
                                              ├─ logistic / random forest / gradient boost
                                              └─ metrics + Streamlit operations dashboard
```

The implementation is developed in milestone commits so the design history is reviewable.

