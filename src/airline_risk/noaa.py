"""NOAA Climate Data Online v2 client and daily-weather normalization."""

from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
WEATHER_TYPES = ("TMAX", "TMIN", "PRCP", "SNOW", "AWND")


class NoaaApiError(RuntimeError):
    """Raised for retryable or malformed NOAA responses."""


class NoaaClient:
    """Token-authenticated client respecting NOAA's five requests/second limit."""

    def __init__(
        self,
        token: str,
        transport: httpx.BaseTransport | None = None,
        requests_per_second: float = 4.0,
    ):
        if not token or token == "replace-with-your-noaa-token":
            raise ValueError("Set NOAA_TOKEN to a valid Climate Data Online token")
        self._minimum_interval = 1 / min(requests_per_second, 5.0)
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._client = httpx.Client(
            base_url=NOAA_BASE_URL,
            headers={"token": token, "Accept": "application/json"},
            timeout=60,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def _throttle(self) -> None:
        with self._lock:
            wait_for = self._minimum_interval - (time.monotonic() - self._last_request)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_request = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, NoaaApiError)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, endpoint: str, params: list[tuple[str, str | int]]) -> dict[str, Any]:
        self._throttle()
        response = self._client.get(endpoint, params=params)
        if response.status_code in {429, 500, 502, 503, 504}:
            raise NoaaApiError(f"Retryable NOAA response {response.status_code}")
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise NoaaApiError("NOAA returned invalid JSON") from exc

    def daily_data(self, station_id: str, start: date, end: date) -> list[dict[str, Any]]:
        """Fetch all GHCND observations for one station/date range with pagination."""

        common: list[tuple[str, str | int]] = [
            ("datasetid", "GHCND"),
            ("stationid", station_id),
            ("startdate", start.isoformat()),
            ("enddate", end.isoformat()),
            ("units", "standard"),
            ("includemetadata", "true"),
            ("limit", 1000),
        ]
        common.extend(("datatypeid", value) for value in WEATHER_TYPES)
        observations: list[dict[str, Any]] = []
        offset = 1
        while True:
            payload = self._get("/data", [*common, ("offset", offset)])
            observations.extend(payload.get("results", []))
            count = payload.get("metadata", {}).get("resultset", {}).get("count", 0)
            if offset + 1000 > count:
                break
            offset += 1000
        return observations


def yearly_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        boundary = min(date(cursor.year, 12, 31), end)
        ranges.append((cursor, boundary))
        cursor = boundary + timedelta(days=1)
    return ranges


def normalize_weather(observations: list[dict[str, Any]], airport: str) -> pd.DataFrame:
    if not observations:
        return pd.DataFrame(
            columns=["airport", "weather_date", *[value.lower() for value in WEATHER_TYPES]]
        )
    weather = pd.DataFrame(observations)
    required = {"date", "datatype", "value"}
    if missing := required - set(weather.columns):
        raise ValueError(f"NOAA response missing required fields: {sorted(missing)}")
    weather["weather_date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather["value"] = pd.to_numeric(weather["value"], errors="coerce")
    weather = weather[weather["datatype"].isin(WEATHER_TYPES)]
    wide = weather.pivot_table(
        index="weather_date", columns="datatype", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={value: value.lower() for value in WEATHER_TYPES})
    for column in (value.lower() for value in WEATHER_TYPES):
        if column not in wide:
            wide[column] = pd.NA
    wide.insert(0, "airport", airport)
    return wide[["airport", "weather_date", *[value.lower() for value in WEATHER_TYPES]]]


def fetch_weather_universe(
    stations: pd.DataFrame,
    start: date,
    end: date,
    token: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """Fetch/cache every airport-year response and return a tidy daily table."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    with NoaaClient(token) as client:
        for station in stations.itertuples(index=False):
            airport_observations: list[dict[str, Any]] = []
            for range_start, range_end in yearly_ranges(start, end):
                cache = raw_dir / f"{station.airport}_{range_start}_{range_end}.json"
                if cache.exists():
                    observations = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    observations = client.daily_data(
                        station.station_id, range_start, range_end
                    )
                    cache.write_text(json.dumps(observations), encoding="utf-8")
                airport_observations.extend(observations)
            frames.append(normalize_weather(airport_observations, station.airport))
    return pd.concat(frames, ignore_index=True)

