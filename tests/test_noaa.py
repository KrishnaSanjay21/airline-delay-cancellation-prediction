from datetime import date

import httpx

from airline_risk.noaa import NoaaClient, normalize_weather, yearly_ranges


def test_yearly_ranges_split_calendar_years() -> None:
    assert yearly_ranges(date(2023, 12, 30), date(2024, 1, 2)) == [
        (date(2023, 12, 30), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 1, 2)),
    ]


def test_noaa_client_sends_token_and_parses_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["token"] == "valid-token"
        return httpx.Response(
            200,
            json={
                "metadata": {"resultset": {"count": 1}},
                "results": [{"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 50}],
            },
        )

    with NoaaClient("valid-token", transport=httpx.MockTransport(handler)) as client:
        results = client.daily_data("GHCND:TEST", date(2024, 1, 1), date(2024, 1, 1))
    assert results[0]["value"] == 50


def test_normalize_weather_pivots_and_fills_missing_types() -> None:
    result = normalize_weather(
        [
            {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 50},
            {"date": "2024-01-01T00:00:00", "datatype": "PRCP", "value": 0.2},
        ],
        "ATL",
    )
    assert result.loc[0, "airport"] == "ATL"
    assert result.loc[0, "tmax"] == 50
    assert "snow" in result.columns


def test_noaa_client_requires_non_placeholder_token() -> None:
    try:
        NoaaClient("replace-with-your-noaa-token")
    except ValueError as exc:
        assert "NOAA_TOKEN" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

