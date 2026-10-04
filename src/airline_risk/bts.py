"""Download and normalize BTS on-time performance archives."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

BTS_PREZIP = "https://transtats.bts.gov/PREZIP"

KEEP_COLUMNS = {
    "FlightDate",
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "Marketing_Airline_Network",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDelay",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "CancellationCode",
}


def month_starts(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def archive_candidates(year: int, month: int, dataset: str = "reporting") -> list[str]:
    if dataset == "marketing":
        names = [
            f"On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
        ]
    else:
        names = [
            f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip",
            f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip",
        ]
    return [f"{BTS_PREZIP}/{name}" for name in names]


@retry(wait=wait_exponential(min=1, max=15), stop=stop_after_attempt(3), reraise=True)
def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(".part")
    with httpx.stream(
        "GET",
        url,
        timeout=120,
        follow_redirects=True,
        headers={"User-Agent": "airline-risk-portfolio/0.1"},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"BTS response was not a ZIP archive: {url}")
    shutil.move(temporary, destination)


def download_month(year: int, month: int, output_dir: Path, dataset: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"bts_{dataset}_{year}_{month:02d}.zip"
    if destination.exists() and zipfile.is_zipfile(destination):
        return destination

    last_error: Exception | None = None
    for url in archive_candidates(year, month, dataset):
        try:
            _download(url, destination)
            return destination
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code != 404:
                raise
    raise FileNotFoundError(f"No BTS archive found for {year}-{month:02d}") from last_error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bts_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {path.name}; found {len(csv_names)}")
        with archive.open(csv_names[0]) as source:
            flights = pd.read_csv(
                source,
                usecols=lambda column: column.strip() in KEEP_COLUMNS,
                low_memory=False,
            )

    flights.columns = [column.strip() for column in flights.columns]
    if "Reporting_Airline" not in flights and "Marketing_Airline_Network" in flights:
        flights["Reporting_Airline"] = flights["Marketing_Airline_Network"]
    required = {"FlightDate", "Reporting_Airline", "Origin", "Dest", "ArrDel15", "Cancelled"}
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"BTS archive missing required columns: {sorted(missing)}")
    flights["FlightDate"] = pd.to_datetime(flights["FlightDate"], errors="coerce")
    return flights


def load_bts_range(
    start: date,
    end: date,
    output_dir: Path,
    dataset: str = "reporting",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year, month in month_starts(start, end):
        archive = download_month(year, month, output_dir, dataset)
        frame = read_bts_zip(archive)
        frame["source_archive"] = archive.name
        frame["source_sha256"] = file_sha256(archive)
        frames.append(frame)
    flights = pd.concat(frames, ignore_index=True)
    mask = flights["FlightDate"].dt.date.between(start, end)
    return flights.loc[mask].reset_index(drop=True)

