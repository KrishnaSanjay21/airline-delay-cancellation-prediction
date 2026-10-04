import numpy as np
import pandas as pd

from airline_risk.evaluation import binary_metrics, calibration_table


def test_binary_metrics_are_computed() -> None:
    metrics = binary_metrics(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["true_positive"] == 2


def test_calibration_table_has_probability_pairs() -> None:
    table = calibration_table([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9]), bins=2)
    assert list(table.columns) == ["mean_predicted_probability", "observed_rate"]
    assert len(table) == 2

