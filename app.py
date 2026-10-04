"""Streamlit operations and model-validation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "artifacts" / "metrics"

st.set_page_config(page_title="AirOps Risk Monitor", page_icon="✈️", layout="wide")


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / name)


@st.cache_data
def read_predictions(target: str) -> pd.DataFrame:
    frame = pd.read_parquet(METRICS / f"{target}_test_predictions.parquet")
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"])
    return frame


@st.cache_data
def read_manifest() -> dict:
    return json.loads((METRICS / "model_manifest.json").read_text(encoding="utf-8"))


def artifact_ready() -> bool:
    required = [
        "model_manifest.json",
        "delay_target_test_predictions.parquet",
        "cancel_target_test_predictions.parquet",
    ]
    return all((METRICS / name).exists() for name in required)


st.title("AirOps Risk Monitor")
st.caption(
    "BTS on-time performance joined with NOAA airport weather · "
    "chronological holdout evaluation"
)

if not artifact_ready():
    st.error("Model artifacts have not been created on this environment.")
    st.code("airline-risk run-all", language="bash")
    st.info("Set NOAA_TOKEN and the date range in .env before running the pipeline.")
    st.stop()

manifest = read_manifest()
label = st.sidebar.radio("Prediction target", ["Arrival delay ≥15 min", "Cancellation"])
target = "delay_target" if label.startswith("Arrival") else "cancel_target"
predictions = read_predictions(target)

airports = sorted(predictions["Origin"].dropna().unique())
carriers = sorted(predictions["Reporting_Airline"].dropna().unique())
selected_airports = st.sidebar.multiselect("Origin airports", airports)
selected_carriers = st.sidebar.multiselect("Carriers", carriers)
filtered = predictions.copy()
if selected_airports:
    filtered = filtered[filtered["Origin"].isin(selected_airports)]
if selected_carriers:
    filtered = filtered[filtered["Reporting_Airline"].isin(selected_carriers)]

overview, validation, airport_tab, records = st.tabs(
    ["Operations overview", "Model validation", "Airport performance", "Scored flights"]
)

with overview:
    positive_rate = filtered[target].mean() if len(filtered) else 0
    predicted_rate = filtered["risk_probability"].mean() if len(filtered) else 0
    high_risk = (filtered["risk_probability"] >= 0.5).mean() if len(filtered) else 0
    columns = st.columns(4)
    columns[0].metric("Holdout flights", f"{len(filtered):,}")
    columns[1].metric("Observed disruption rate", f"{positive_rate:.1%}")
    columns[2].metric("Mean predicted risk", f"{predicted_rate:.1%}")
    columns[3].metric("Flights above 50% risk", f"{high_risk:.1%}")

    daily = (
        filtered.groupby("FlightDate", as_index=False)
        .agg(observed_rate=(target, "mean"), predicted_rate=("risk_probability", "mean"))
        .melt("FlightDate", var_name="series", value_name="rate")
    )
    st.plotly_chart(
        px.line(
            daily,
            x="FlightDate",
            y="rate",
            color="series",
            title="Observed and predicted disruption rate over the holdout period",
            labels={"rate": "Rate", "series": ""},
        ),
        use_container_width=True,
    )

    by_origin = (
        filtered.groupby("Origin", as_index=False)
        .agg(
            flights=("flight_id", "count"),
            observed_rate=(target, "mean"),
            predicted_rate=("risk_probability", "mean"),
        )
        .query("flights >= 25")
        .sort_values("predicted_rate", ascending=False)
    )
    st.plotly_chart(
        px.bar(
            by_origin,
            x="Origin",
            y="predicted_rate",
            color="observed_rate",
            hover_data=["flights"],
            title="Risk by origin airport",
            labels={"predicted_rate": "Mean predicted risk", "observed_rate": "Observed rate"},
        ),
        use_container_width=True,
    )

with validation:
    comparison = read_csv(f"{target}_model_comparison.csv")
    st.subheader("Candidate model comparison")
    st.dataframe(
        comparison[
            [
                "model",
                "roc_auc",
                "average_precision",
                "brier_score",
                "precision_at_0_5",
                "recall_at_0_5",
                "f1_at_0_5",
            ]
        ].style.format(precision=3),
        use_container_width=True,
        hide_index=True,
    )

    calibration = read_csv(f"{target}_calibration.csv")
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["observed_rate"],
            mode="lines+markers",
            name="Selected model",
        )
    )
    chart.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration"))
    chart.update_layout(
        title="Calibration on untouched chronological holdout",
        xaxis_title="Predicted probability",
        yaxis_title="Observed rate",
    )
    st.plotly_chart(chart, use_container_width=True)

    importance = read_csv(f"{target}_feature_importance.csv").head(15).sort_values(
        "importance_mean"
    )
    st.plotly_chart(
        px.bar(
            importance,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="Permutation importance (average-precision decrease)",
        ),
        use_container_width=True,
    )
    split = manifest["split"]
    st.caption(
        f"Train: {split['train_start']}–{split['train_end']} · "
        f"Calibration: {split['calibration_start']}–{split['calibration_end']} · "
        f"Test: {split['test_start']}–{split['test_end']}"
    )

with airport_tab:
    airport_metrics = read_csv(f"{target}_airport_performance.csv")
    st.dataframe(airport_metrics, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.scatter(
            airport_metrics,
            x="positive_rate",
            y="average_precision",
            size="rows",
            color="roc_auc",
            hover_name="Origin",
            title="Airport discrimination relative to disruption prevalence",
        ),
        use_container_width=True,
    )

with records:
    threshold = st.slider("Risk threshold", 0.05, 0.95, 0.50, 0.05)
    display = filtered[filtered["risk_probability"] >= threshold].sort_values(
        "risk_probability", ascending=False
    )
    st.dataframe(
        display[
            [
                "FlightDate",
                "Reporting_Airline",
                "Origin",
                "Dest",
                target,
                "risk_probability",
                "best_model",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Source: U.S. DOT BTS and NOAA NCEI. Portfolio demonstration—not operational guidance. "
    "The dashboard displays holdout predictions, not live flight forecasts."
)

