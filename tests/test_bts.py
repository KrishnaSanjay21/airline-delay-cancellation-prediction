import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from airline_risk.bts import archive_candidates, month_starts, read_bts_zip


def test_month_starts_crosses_year_boundary() -> None:
    assert list(month_starts(date(2023, 12, 1), date(2024, 2, 2))) == [
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_archive_candidates_cover_bts_filename_variants() -> None:
    reporting = archive_candidates(2024, 1, "reporting")
    marketing = archive_candidates(2024, 1, "marketing")
    assert len(reporting) == 2
    assert "(1987_present)" in reporting[0]
    assert "Marketing" in marketing[0]


def test_read_bts_zip_normalizes_required_columns(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "FlightDate": ["2024-01-01"],
            "Reporting_Airline": ["AA"],
            "Origin": ["DFW"],
            "Dest": ["ATL"],
            "CRSDepTime": [900],
            "CRSArrTime": [1200],
            "CRSElapsedTime": [120],
            "Distance": [700],
            "ArrDel15": [0],
            "Cancelled": [0],
        }
    )
    csv_path = tmp_path / "flights.csv"
    data.to_csv(csv_path, index=False)
    zip_path = tmp_path / "flights.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="flights.csv")
    result = read_bts_zip(zip_path)
    assert result.loc[0, "Origin"] == "DFW"
    assert str(result["FlightDate"].dtype).startswith("datetime64")

