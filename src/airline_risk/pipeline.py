"""Reproducible ingestion, feature, training, and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .bts import load_bts_range
from .config import Settings, ensure_directories, load_airport_stations
from .evaluation import (
    airport_performance,
    calibration_table,
    evaluate_candidates,
    permutation_importance,
    select_best_model,
)
from .features import build_feature_table, quality_report
from .modeling import train_candidates
from .noaa import fetch_weather_universe
from .validation import (
    assert_leakage_safe,
    assert_split_order,
    chronological_split,
    split_for_target,
)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def ingest_sources(settings: Settings) -> tuple[Path, Path]:
    ensure_directories(settings)
    if not settings.noaa_token:
        raise ValueError("NOAA_TOKEN is required for source ingestion")
    stations = load_airport_stations(settings.top_airports)
    flights = load_bts_range(
        settings.start_date,
        settings.end_date,
        settings.raw_dir / "bts",
        settings.bts_dataset,
    )
    weather = fetch_weather_universe(
        stations,
        settings.start_date,
        settings.end_date,
        settings.noaa_token,
        settings.raw_dir / "noaa",
    )
    flights_path = settings.processed_dir / "flights.parquet"
    weather_path = settings.processed_dir / "weather.parquet"
    flights.to_parquet(flights_path, index=False)
    weather.to_parquet(weather_path, index=False)
    return flights_path, weather_path


def build_dataset(settings: Settings) -> Path:
    stations = load_airport_stations(settings.top_airports)
    supported = set(stations["airport"])
    flights = pd.read_parquet(settings.processed_dir / "flights.parquet")
    weather = pd.read_parquet(settings.processed_dir / "weather.parquet")
    flights = flights[
        flights["Origin"].isin(supported) & flights["Dest"].isin(supported)
    ].copy()
    features = build_feature_table(flights, weather)
    report = quality_report(features)
    if report["duplicate_flight_ids"]:
        raise ValueError(
            f"Found {report['duplicate_flight_ids']} duplicate flight identifiers; review source"
        )
    if report["invalid_airport_codes"] or report["cancelled_with_delay_target"]:
        raise ValueError(f"Blocking data-quality failures: {report}")
    output = settings.processed_dir / "model_features.parquet"
    features.to_parquet(output, index=False)
    _write_json(report, settings.artifacts_dir / "metrics" / "data_quality.json")
    return output


def _save_target_outputs(
    settings: Settings,
    target: str,
    test: pd.DataFrame,
    candidates: dict,
    comparison: pd.DataFrame,
    probabilities: dict,
) -> dict:
    metrics_dir = settings.artifacts_dir / "metrics"
    models_dir = settings.artifacts_dir / "models"
    best_name = select_best_model(comparison)
    best_model = candidates[best_name]
    best_probability = probabilities[best_name]

    comparison.to_csv(metrics_dir / f"{target}_model_comparison.csv", index=False)
    calibration_table(test[target], best_probability).to_csv(
        metrics_dir / f"{target}_calibration.csv", index=False
    )
    airport_performance(test, best_probability, target, minimum_rows=25).to_csv(
        metrics_dir / f"{target}_airport_performance.csv", index=False
    )
    permutation_importance(best_model, test, target).to_csv(
        metrics_dir / f"{target}_feature_importance.csv", index=False
    )

    scored_columns = [
        "flight_id",
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        target,
    ]
    scored = test[scored_columns].copy()
    for name, probability in probabilities.items():
        scored[f"{name}_probability"] = probability
    scored["best_model"] = best_name
    scored["risk_probability"] = best_probability
    scored.to_parquet(metrics_dir / f"{target}_test_predictions.parquet", index=False)
    joblib.dump(best_model, models_dir / f"{target}_best_model.joblib")

    best_row = comparison.iloc[0].to_dict()
    return {"target": target, "best_model": best_name, "holdout_metrics": best_row}


def train_and_evaluate(settings: Settings) -> dict:
    ensure_directories(settings)
    assert_leakage_safe()
    features = pd.read_parquet(settings.processed_dir / "model_features.parquet")
    split = chronological_split(features)
    assert_split_order(split)
    _write_json(
        split.metadata.to_dict(), settings.artifacts_dir / "metrics" / "split_metadata.json"
    )

    results = []
    for target in ("delay_target", "cancel_target"):
        target_split = split_for_target(split, target)
        candidates = train_candidates(
            target_split.train,
            target_split.calibration,
            target,
            seed=settings.random_seed,
        )
        comparison, probabilities = evaluate_candidates(candidates, target_split.test, target)
        results.append(
            _save_target_outputs(
                settings,
                target,
                target_split.test,
                candidates,
                comparison,
                probabilities,
            )
        )

    manifest = {
        "data_period": {"start": str(settings.start_date), "end": str(settings.end_date)},
        "split": split.metadata.to_dict(),
        "models": results,
        "prediction_point": "scheduled departure",
        "calibration_method": "sigmoid scaling on chronological calibration period",
    }
    _write_json(manifest, settings.artifacts_dir / "metrics" / "model_manifest.json")
    return manifest


def run_all(settings: Settings) -> dict:
    ingest_sources(settings)
    build_dataset(settings)
    return train_and_evaluate(settings)

