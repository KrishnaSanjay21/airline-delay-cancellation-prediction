"""Chronological model-development splits and leakage assertions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from .features import LEAKAGE_COLUMNS, MODEL_FEATURES


@dataclass(frozen=True)
class SplitMetadata:
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str
    train_rows: int
    calibration_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    metadata: SplitMetadata


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split on whole dates, never randomly, keeping the final period untouched."""

    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("Train and calibration fractions must leave a test period")

    ordered = frame.sort_values(["FlightDate", "flight_id"]).copy()
    ordered["FlightDate"] = pd.to_datetime(ordered["FlightDate"]).dt.normalize()
    unique_dates = ordered["FlightDate"].dropna().drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 7:
        raise ValueError("At least seven distinct flight dates are required")

    train_index = max(1, int(len(unique_dates) * train_fraction))
    calibration_index = max(train_index + 1, int(len(unique_dates) * (train_fraction + calibration_fraction)))
    calibration_index = min(calibration_index, len(unique_dates) - 1)
    train_end = unique_dates[train_index - 1]
    calibration_end = unique_dates[calibration_index - 1]

    train = ordered[ordered["FlightDate"] <= train_end].copy()
    calibration = ordered[
        (ordered["FlightDate"] > train_end) & (ordered["FlightDate"] <= calibration_end)
    ].copy()
    test = ordered[ordered["FlightDate"] > calibration_end].copy()

    metadata = SplitMetadata(
        train_start=train["FlightDate"].min().date().isoformat(),
        train_end=train["FlightDate"].max().date().isoformat(),
        calibration_start=calibration["FlightDate"].min().date().isoformat(),
        calibration_end=calibration["FlightDate"].max().date().isoformat(),
        test_start=test["FlightDate"].min().date().isoformat(),
        test_end=test["FlightDate"].max().date().isoformat(),
        train_rows=len(train),
        calibration_rows=len(calibration),
        test_rows=len(test),
    )
    return ChronologicalSplit(train, calibration, test, metadata)


def assert_leakage_safe(feature_columns: list[str] | None = None) -> None:
    columns = feature_columns or MODEL_FEATURES
    leaked = LEAKAGE_COLUMNS.intersection(columns)
    if leaked:
        raise ValueError(f"Post-outcome columns cannot be model features: {sorted(leaked)}")


def assert_split_order(split: ChronologicalSplit) -> None:
    train_max = pd.to_datetime(split.train["FlightDate"]).max()
    calibration_min = pd.to_datetime(split.calibration["FlightDate"]).min()
    calibration_max = pd.to_datetime(split.calibration["FlightDate"]).max()
    test_min = pd.to_datetime(split.test["FlightDate"]).min()
    if not train_max < calibration_min <= calibration_max < test_min:
        raise AssertionError("Chronological split periods overlap or are out of order")


def split_for_target(split: ChronologicalSplit, target: str) -> ChronologicalSplit:
    """Remove rows whose target is structurally unavailable (cancelled delay cases)."""

    if target not in {"delay_target", "cancel_target"}:
        raise ValueError("Unknown target")
    train = split.train.dropna(subset=[target]).copy()
    calibration = split.calibration.dropna(subset=[target]).copy()
    test = split.test.dropna(subset=[target]).copy()
    return ChronologicalSplit(train, calibration, test, split.metadata)

