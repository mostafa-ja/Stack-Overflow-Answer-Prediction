"""
Tests for evaluation metrics and plotting utilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    perform_error_analysis,
)

def test_compute_all_metrics():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.95, 0.05])

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0

def test_plots(tmp_path):
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.95, 0.05])

    cm_path = tmp_path / "cm.png"
    fig_cm = plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    assert cm_path.exists()

    roc_path = tmp_path / "roc.png"
    fig_roc = plot_roc_curve(y_true, y_prob, save_path=roc_path)
    assert roc_path.exists()

    fi_series = pd.Series({"FeatureA": 0.5, "FeatureB": 0.3, "FeatureC": 0.2})
    fi_path = tmp_path / "fi.png"
    fig_fi = plot_feature_importance(fi_series, save_path=fi_path)
    assert fi_path.exists()

def test_error_analysis():
    X = pd.DataFrame({
        "Reputation": [1000, 50, 500, 200],
        "Time_Diffrence(H)": [0.5, 10.0, 1.2, 5.0],
    })
    y_test = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # One FP (idx 1), one FN (idx 2)
    y_prob = np.array([0.8, 0.7, 0.3, 0.2])

    diag = perform_error_analysis(X, y_test, y_pred, y_prob)
    assert diag["false_positive_count"] == 1
    assert diag["false_negative_count"] == 1
