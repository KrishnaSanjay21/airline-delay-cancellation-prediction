"""Candidate classifiers and held-out sigmoid probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        sparse_threshold=0,
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    """Return three intentionally different, class-balanced model families."""

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _linear_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _tree_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", _tree_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=200,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class CalibratedCandidate:
    """A fitted base model plus Platt scaling learned only on the next time period."""

    name: str
    target: str
    base_model: ClassifierMixin
    calibrator: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw_probability = self.base_model.predict_proba(frame[MODEL_FEATURES])[:, 1]
        positive = self.calibrator.predict_proba(_logit(raw_probability))[:, 1]
        return np.column_stack([1 - positive, positive])

    def predict(self, frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= threshold).astype(int)


def train_candidates(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    target: str,
    seed: int = 42,
) -> dict[str, CalibratedCandidate]:
    if target not in {"delay_target", "cancel_target"}:
        raise ValueError("Unknown target")
    y_train = train[target].astype(int)
    y_calibration = calibration[target].astype(int)
    if y_train.nunique() < 2 or y_calibration.nunique() < 2:
        raise ValueError(f"Both classes are required in train and calibration sets for {target}")

    fitted: dict[str, CalibratedCandidate] = {}
    for name, model in candidate_models(seed).items():
        model.fit(train[MODEL_FEATURES], y_train)
        raw_probability = model.predict_proba(calibration[MODEL_FEATURES])[:, 1]
        calibrator = LogisticRegression(random_state=seed)
        calibrator.fit(_logit(raw_probability), y_calibration)
        fitted[name] = CalibratedCandidate(name, target, model, calibrator)
    return fitted

