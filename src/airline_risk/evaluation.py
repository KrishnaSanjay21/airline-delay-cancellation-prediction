"""Holdout metrics, calibration diagnostics, and model-agnostic explainability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import MODEL_FEATURES
from .modeling import CalibratedCandidate


def binary_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None
    average_precision = (
        float(average_precision_score(y, probability)) if np.unique(y).size == 2 else None
    )
    return {
        "rows": len(y),
        "positive_rate": float(y.mean()),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(y, probability)),
        "precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, predicted, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def calibration_table(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    observed, predicted = calibration_curve(
        np.asarray(y_true, dtype=int), probability, n_bins=bins, strategy="quantile"
    )
    return pd.DataFrame({"mean_predicted_probability": predicted, "observed_rate": observed})


def evaluate_candidates(
    candidates: Mapping[str, CalibratedCandidate],
    test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        probability = model.predict_proba(test)[:, 1]
        probabilities[name] = probability
        rows.append({"model": name, **binary_metrics(test[target], probability)})
    comparison = pd.DataFrame(rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False, na_position="last"
    )
    return comparison.reset_index(drop=True), probabilities


def airport_performance(
    frame: pd.DataFrame,
    probability: np.ndarray,
    target: str,
    airport_column: str = "Origin",
    minimum_rows: int = 100,
) -> pd.DataFrame:
    scored = frame[[airport_column, target]].copy()
    scored["probability"] = probability
    rows = []
    for airport, group in scored.groupby(airport_column):
        if len(group) < minimum_rows:
            continue
        rows.append({airport_column: airport, **binary_metrics(group[target], group["probability"])})
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def permutation_importance(
    model: CalibratedCandidate,
    frame: pd.DataFrame,
    target: str,
    repeats: int = 3,
    max_rows: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure average-precision loss when each input feature is independently shuffled."""

    sample = frame.sample(min(len(frame), max_rows), random_state=seed).copy()
    y_true = sample[target].astype(int)
    baseline = average_precision_score(y_true, model.predict_proba(sample)[:, 1])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for feature in MODEL_FEATURES:
        losses = []
        original = sample[feature].copy()
        for _ in range(repeats):
            sample[feature] = rng.permutation(original.to_numpy())
            score = average_precision_score(y_true, model.predict_proba(sample)[:, 1])
            losses.append(baseline - score)
        sample[feature] = original
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(losses)),
                "importance_std": float(np.std(losses)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def select_best_model(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        raise ValueError("No evaluated models")
    return str(comparison.iloc[0]["model"])

