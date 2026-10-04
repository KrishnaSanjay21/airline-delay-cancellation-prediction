"""Command line entry points for data and model pipelines."""

from __future__ import annotations

import json

import typer

from .config import Settings
from .pipeline import build_dataset, ingest_sources, run_all, train_and_evaluate

app = typer.Typer(no_args_is_help=True, help="BTS + NOAA airline disruption pipeline")


@app.command("ingest")
def ingest_command() -> None:
    flights, weather = ingest_sources(Settings())
    typer.echo(f"Flights: {flights}\nWeather: {weather}")


@app.command("build-features")
def build_features_command() -> None:
    typer.echo(build_dataset(Settings()))


@app.command("train")
def train_command() -> None:
    typer.echo(json.dumps(train_and_evaluate(Settings()), indent=2, default=str))


@app.command("run-all")
def run_all_command() -> None:
    typer.echo(json.dumps(run_all(Settings()), indent=2, default=str))


if __name__ == "__main__":
    app()

