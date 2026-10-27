"""
Tests for data loading, synthetic data generation, and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    generate_synthetic_so_dataset,
    compute_leave_one_out_ratio,
    load_dataset,
    get_train_val_test_splits,
)
from src.config import FEATURE_COLS_REALTIME, FEATURE_COLS_FULL, TARGET_COL

def test_synthetic_data_generation():
    df = generate_synthetic_so_dataset(n_samples=500, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert TARGET_COL in df.columns
    # Check that labels are binary
    assert set(df[TARGET_COL].unique()).issubset({0, 1})
    # Ensure key features exist
    for col in ["Time_Diffrence(H)", "Earlier_Answers", "Reputation", "badgeCount"]:
        assert col in df.columns

def test_leave_one_out_ratio():
    sample_df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "accepted": [1, 0, 1, 0, 0, 1],
    })
    loo = compute_leave_one_out_ratio(sample_df, group_col="user_id", target_col="accepted", min_count=2)
    # User 1 has 3 answers: sum=2.
    # Row 0: (2 - 1) / (3 - 1) = 0.5
    # Row 1: (2 - 0) / (3 - 1) = 1.0
    # Row 2: (2 - 1) / (3 - 1) = 0.5
    assert np.isclose(loo.iloc[0], 0.5)
    assert np.isclose(loo.iloc[1], 1.0)
    assert np.isclose(loo.iloc[2], 0.5)
    # User 3 has count=1 < min_count=2 -> imputed with mean
    assert not np.isnan(loo.iloc[5])

def test_load_dataset_realtime():
    X, y = load_dataset(feature_set="realtime", n_samples=300, allow_synthetic_fallback=True)
    assert len(X) == 300
    assert len(y) == 300
    # Zero leakage: ensure post-hoc answer features are not in X
    assert "Score_A" not in X.columns
    assert "CommentCount_A" not in X.columns

def test_load_dataset_full():
    X, y = load_dataset(feature_set="full", n_samples=300, allow_synthetic_fallback=True)
    assert len(X) == 300
    assert "Score_A" in X.columns

def test_train_val_test_splits():
    df = generate_synthetic_so_dataset(n_samples=400, random_state=42)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(X, y, test_size=0.2, val_size=0.1)
    assert len(X_train) + len(X_val) + len(X_test) == 400
    assert len(X_test) == 80  # 20%
    assert len(X_val) == 40   # 10%
    assert len(X_train) == 280
