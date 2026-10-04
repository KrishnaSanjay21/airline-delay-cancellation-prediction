# Operations runbook

## First run

1. Request a free NOAA Climate Data Online token.
2. Copy `.env.example` to `.env` and set the token.
3. Start with a three-month period already present in the BTS PREZIP directory.
4. Run `airline-risk run-all`.
5. Review `artifacts/metrics/data_quality.json` and both model-comparison files.
6. Start the dashboard with `streamlit run app.py`.

## Common failures

### BTS archive not found

BTS publishes with a reporting lag, so a recent calendar month may not exist yet. Verify the month in the official PREZIP directory and select an available period. The downloader supports both common reporting-carrier filename variants.

### NOAA 401 or 403

Check the token in `.env`. The token belongs in the `token` request header, not in a public repository. Never commit it.

### NOAA 429

The client limits itself below five requests per second and retries transient failures. Stop parallel runs and wait before retrying. Cached airport-year responses reduce repeated calls.

### Duplicate flight identifiers

The feature build stops instead of silently dropping rows. Inspect carrier, flight number, scheduled time, route, and BTS duplicate indicators, then document any deduplication rule before implementation.

### One class missing from calibration

Cancellation is uncommon. Expand the date range until both outcomes appear in train, calibration, and test periods; do not replace the chronological split with a random split.

### Streamlit says artifacts are missing

Run the pipeline first or deploy precomputed real-data artifacts from a controlled artifact store. The repository intentionally does not manufacture synthetic model results.

## Deployment distinction

The included Docker and workflow files are implementation assets. Do not claim a live, scheduled, or production model until data credentials, artifact persistence, dashboard hosting, monitoring, and successful refresh history are verified in the target environment.

