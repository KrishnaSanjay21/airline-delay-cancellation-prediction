# GitHub Actions templates

The `.example` suffix prevents accidental activation. To enable these workflows, copy them to `.github/workflows/` using a GitHub credential with the `workflow` scope.

The refresh workflow is manual by design because BTS monthly files are published with a reporting lag. Supply a period already available in the BTS PREZIP directory, add `NOAA_TOKEN` as a repository secret, and inspect the uploaded evaluation artifacts before deploying a new model.

