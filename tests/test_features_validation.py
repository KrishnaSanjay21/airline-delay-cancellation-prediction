import numpy as np
import pandas as pd
import pytest

from airline_risk.features import MODEL_FEATURES, build_feature_table, quality_report
from airline_risk.validation import (
    assert_leakage_safe,
    assert_split_order,
    chronological_split,
    split_for_target,
)


def sample_features(days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days)
    flights = pd.DataFrame(
        {
            "FlightDate": dates,
            "Reporting_Airline": ["AA", "DL"] * (days // 2),
            "Flight_Number_Reporting_Airline": range(days),
            "Origin": ["ATL"] * days,
            "Dest": ["DFW"] * days,
            "CRSDepTime": [900] * days,
            "CRSArrTime": [1100] * days,
            "CRSElapsedTime": [120] * days,
            "Distance": [700] * days,
            "ArrDel15": [index % 2 for index in range(days)],
            "Cancelled": [1 if index == 5 else 0 for index in range(days)],
        }
    )
    weather = pd.concat(
        [
            pd.DataFrame(
                {
                    "airport": airport,
                    "weather_date": dates,
                    "tmax": np.linspace(40, 60, days),
                    "tmin": np.linspace(25, 45, days),
                    "prcp": 0.0,
                    "snow": 0.0,
                    "awnd": 8.0,
                }
            )
            for airport in ("ATL", "DFW")
        ],
        ignore_index=True,
    )
    return build_feature_table(flights, weather)


def test_build_features_joins_both_airports_and_masks_cancelled_delay() -> None:
    result = sample_features()
    assert result["origin_tmax"].notna().all()
    assert result["dest_tmax"].notna().all()
    assert result.loc[result["cancel_target"] == 1, "delay_target"].isna().all()
    assert set(MODEL_FEATURES).issubset(result.columns)


def test_quality_report_has_expected_rates() -> None:
    report = quality_report(sample_features())
    assert report["rows"] == 20
    assert report["duplicate_flight_ids"] == 0
    assert report["weather_missing_rate"] == 0


def test_chronological_split_has_strictly_ordered_periods() -> None:
    split = chronological_split(sample_features())
    assert_split_order(split)
    assert split.train["FlightDate"].max() < split.calibration["FlightDate"].min()
    assert split.calibration["FlightDate"].max() < split.test["FlightDate"].min()


def test_delay_split_removes_cancelled_row() -> None:
    split = split_for_target(chronological_split(sample_features()), "delay_target")
    combined = pd.concat([split.train, split.calibration, split.test])
    assert combined["delay_target"].notna().all()


def test_leakage_assertion_rejects_actual_delay() -> None:
    assert_leakage_safe()
    with pytest.raises(ValueError):
        assert_leakage_safe([*MODEL_FEATURES, "ArrDelay"])

