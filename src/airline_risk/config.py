"""Environment configuration and project paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    noaa_token: str | None = None
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2024, 3, 31)
    bts_dataset: str = "reporting"
    top_airports: int = Field(default=25, ge=2, le=25)
    random_seed: int = 42
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    @field_validator("bts_dataset")
    @classmethod
    def supported_bts_dataset(cls, value: str) -> str:
        if value not in {"reporting", "marketing"}:
            raise ValueError("BTS_DATASET must be 'reporting' or 'marketing'")
        return value

    @model_validator(mode="after")
    def chronological_range(self) -> Settings:
        if self.end_date < self.start_date:
            raise ValueError("END_DATE must be on or after START_DATE")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_airport_stations(limit: int | None = None) -> pd.DataFrame:
    stations = pd.read_csv(PROJECT_ROOT / "config" / "airport_stations.csv")
    return stations.head(limit) if limit is not None else stations


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.raw_dir / "bts",
        settings.raw_dir / "noaa",
        settings.processed_dir,
        settings.artifacts_dir / "models",
        settings.artifacts_dir / "metrics",
    ):
        path.mkdir(parents=True, exist_ok=True)

