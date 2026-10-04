from datetime import date

import pytest

from airline_risk.config import Settings, load_airport_stations


def test_airport_station_map_has_25_unique_airports() -> None:
    stations = load_airport_stations()
    assert len(stations) == 25
    assert stations["airport"].is_unique
    assert stations["station_id"].str.startswith("GHCND:").all()


def test_station_limit() -> None:
    assert len(load_airport_stations(5)) == 5


def test_settings_reject_reverse_dates() -> None:
    with pytest.raises(ValueError):
        Settings(start_date=date(2024, 2, 1), end_date=date(2024, 1, 1))


def test_settings_reject_unknown_bts_dataset() -> None:
    with pytest.raises(ValueError):
        Settings(bts_dataset="unknown")

