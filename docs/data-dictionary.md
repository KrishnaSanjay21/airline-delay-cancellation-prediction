# Data dictionary

## Identifiers and schedule

| Field | Definition |
|---|---|
| `flight_id` | SHA-1 natural-key fingerprint of date, carrier, flight number, route, and scheduled departure |
| `FlightDate` | Scheduled flight date |
| `Reporting_Airline` | BTS reporting operating carrier code |
| `Origin`, `Dest` | Three-letter airport codes |
| `route` | Origin and destination pair |
| `scheduled_dep_hour` | Scheduled departure as a decimal hour |
| `scheduled_arr_hour` | Scheduled arrival as a decimal hour |
| `scheduled_elapsed_minutes` | Published scheduled elapsed time |
| `distance_miles` | BTS flight distance |

## Weather

Each NOAA variable appears with `origin_` and `dest_` prefixes.

| Suffix | NOAA GHCND element, requested with standard units |
|---|---|
| `tmax` | Daily maximum temperature |
| `tmin` | Daily minimum temperature |
| `prcp` | Daily precipitation |
| `snow` | Daily snowfall |
| `awnd` | Average daily wind speed |

The raw NOAA response remains cached by airport and yearly date range. Missing weather is imputed within the model pipeline and monitored separately; it is never converted to zero before modeling.

## Outcomes

| Field | Definition |
|---|---|
| `delay_target` | BTS `ArrDel15`; 1 when arrival delay is at least 15 minutes. Null for cancelled flights. |
| `cancel_target` | BTS `Cancelled`; 1 when the scheduled flight was cancelled. |
| `risk_probability` | Calibrated probability from the selected candidate model on the test period |

## Prohibited leakage fields

`ArrDelay`, `ArrDel15`, `Cancelled`, `Diverted`, `CancellationCode`, actual departure delay/time, actual elapsed time, wheels-off, and wheels-on fields may define outcomes or post-event reporting, but they cannot be predictors.

