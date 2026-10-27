"""
Tests for baseline and gradient boosting models.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models import get_model, MODEL_REGISTRY
from src.data_loader import generate_synthetic_so_dataset
from src.config import TARGET_COL, FEATURE_COLS_REALTIME

@pytest.fixture
def sample_data():
    df = generate_synthetic_so_dataset(n_samples=250, random_state=42)
    X = df[FEATURE_COLS_REALTIME].copy()
    y = df[TARGET_COL].copy()
    return X, y

@pytest.mark.parametrize("model_name", ["logisticregression", "randomforest", "xgboost", "lightgbm"])
def test_model_training_and_inference(model_name, sample_data, tmp_path):
    X, y = sample_data
    model = get_model(model_name)

    # Fit
    model.fit(X, y)

    # Predict
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict proba
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Serialization test
    save_file = tmp_path / f"{model_name}.joblib"
    model.save(save_file)
    assert save_file.exists()

    # Deserialization test
    loaded_model = get_model(model_name)
    loaded_model.load(save_file)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(preds, loaded_preds)
