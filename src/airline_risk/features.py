"""Leakage-safe flight/weather joins and feature engineering."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

WEATHER_COLUMNS = ["tmax", "tmin", "prcp", "snow", "awnd"]
CATEGORICAL_FEATURES = ["Reporting_Airline", "Origin", "Dest", "route"]
NUMERIC_FEATURES = [
    "scheduled_dep_hour",
    "scheduled_arr_hour",
    "scheduled_elapsed_minutes",
    "distance_miles",
    "month",
    "day_of_week",
    "day_of_year_sin",
    "day_of_year_cos",
    *[f"origin_{column}" for column in WEATHER_COLUMNS],
    *[f"dest_{column}" for column in WEATHER_COLUMNS],
]
MODEL_FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]

# These columns exist only after operations unfold and must never enter a pre-departure model.
LEAKAGE_COLUMNS = {
    "ArrDelay",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "CancellationCode",
    "DepDelay",
    "DepDel15",
    "ActualElapsedTime",
    "WheelsOff",
    "WheelsOn",
}


def _hour_from_hhmm(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    hours = np.floor(numeric / 100).where(numeric < 2400, 0)
    minutes = numeric.mod(100)
    valid = hours.between(0, 23) & minutes.between(0, 59)
    return (hours + minutes / 60).where(valid)


def _flight_identifier(frame: pd.DataFrame) -> pd.Series:
    number = frame.get("Flight_Number_Reporting_Airline", pd.Series("", index=frame.index))
    raw = (
        frame["FlightDate"].astype(str)
        + "|"
        + frame["Reporting_Airline"].astype(str)
        + "|"
        + number.astype(str)
        + "|"
        + frame["Origin"].astype(str)
        + "|"
        + frame["Dest"].astype(str)
        + "|"
        + frame.get("CRSDepTime", pd.Series("", index=frame.index)).astype(str)
    )
    return raw.map(lambda value: hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest())


def join_airport_weather(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    result = flights.copy()
    result["FlightDate"] = pd.to_datetime(result["FlightDate"]).dt.normalize()
    weather = weather.copy()
    weather["weather_date"] = pd.to_datetime(weather["weather_date"]).dt.normalize()

    for side, airport_column in (("origin", "Origin"), ("dest", "Dest")):
        renamed = weather.rename(
            columns={
                "airport": airport_column,
                "weather_date": "FlightDate",
                **{column: f"{side}_{column}" for column in WEATHER_COLUMNS},
            }
        )
        result = result.merge(
            renamed[[airport_column, "FlightDate", *[f"{side}_{c}" for c in WEATHER_COLUMNS]]],
            on=[airport_column, "FlightDate"],
            how="left",
            validate="many_to_one",
        )
    return result


def build_feature_table(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    frame = join_airport_weather(flights, weather)
    frame["flight_id"] = _flight_identifier(frame)
    frame["route"] = frame["Origin"].astype(str) + "-" + frame["Dest"].astype(str)
    frame["scheduled_dep_hour"] = _hour_from_hhmm(frame["CRSDepTime"])
    frame["scheduled_arr_hour"] = _hour_from_hhmm(frame["CRSArrTime"])
    frame["scheduled_elapsed_minutes"] = pd.to_numeric(
        frame["CRSElapsedTime"], errors="coerce"
    )
    frame["distance_miles"] = pd.to_numeric(frame["Distance"], errors="coerce")
    frame["month"] = frame["FlightDate"].dt.month
    frame["day_of_week"] = frame["FlightDate"].dt.dayofweek + 1
    day_of_year = frame["FlightDate"].dt.dayofyear
    frame["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["cancel_target"] = pd.to_numeric(frame["Cancelled"], errors="coerce").astype("Int64")
    frame["delay_target"] = pd.to_numeric(frame["ArrDel15"], errors="coerce").astype("Int64")
    frame.loc[frame["cancel_target"] == 1, "delay_target"] = pd.NA
    return frame.sort_values(["FlightDate", "flight_id"]).reset_index(drop=True)


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    weather_features = [
        f"{side}_{column}" for side in ("origin", "dest") for column in WEATHER_COLUMNS
    ]
    duplicate_ids = int(frame["flight_id"].duplicated().sum())
    invalid_airports = int(
        (~frame["Origin"].astype(str).str.fullmatch(r"[A-Z]{3}"))
        .fillna(True)
        .sum()
        + (~frame["Dest"].astype(str).str.fullmatch(r"[A-Z]{3}"))
        .fillna(True)
        .sum()
    )
    cancelled_with_delay_target = int(
        ((frame["cancel_target"] == 1) & frame["delay_target"].notna()).sum()
    )
    return {
        "rows": len(frame),
        "date_min": frame["FlightDate"].min().date().isoformat(),
        "date_max": frame["FlightDate"].max().date().isoformat(),
        "duplicate_flight_ids": duplicate_ids,
        "invalid_airport_codes": invalid_airports,
        "cancelled_with_delay_target": cancelled_with_delay_target,
        "weather_missing_rate": float(frame[weather_features].isna().all(axis=1).mean()),
        "delay_rate": float(frame["delay_target"].mean()),
        "cancellation_rate": float(frame["cancel_target"].mean()),
    }

