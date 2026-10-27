"""
Baseline Machine Learning models for Stack Overflow answer prediction.
Includes Logistic Regression, Random Forest, and Multi-Layer Perceptron (MLP).
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel
from src.config import DEFAULT_LOGREG_PARAMS, DEFAULT_RF_PARAMS

class LogisticRegressionModel(BaseModel):
    """L2-regularized Logistic Regression baseline with standard feature scaling."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("LogisticRegression", params or DEFAULT_LOGREG_PARAMS)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(**self.params)
        self.pipeline = Pipeline([
            ("scaler", self.scaler),
            ("classifier", self.classifier),
        ])
        self.model = self.pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LogisticRegressionModel":
        self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        self.model = self.pipeline
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Optional[pd.Series]:
        clf = self.model.named_steps.get("classifier") if hasattr(self.model, "named_steps") else self.model
        if hasattr(clf, "coef_") and self.feature_names:
            coefs = np.abs(clf.coef_[0])
            return pd.Series(coefs, index=self.feature_names).sort_values(ascending=False)
        return None


class RandomForestModel(BaseModel):
    """Random Forest ensemble baseline."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("RandomForest", params or DEFAULT_RF_PARAMS)
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "RandomForestModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class MLPModel(BaseModel):
    """Multi-Layer Perceptron (Neural Network) baseline."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_mlp = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "max_iter": 200,
            "random_state": 42,
            "early_stopping": True,
        }
        merged_params = {**default_mlp, **(params or {})}
        super().__init__("MLPClassifier", merged_params)
        self.scaler = StandardScaler()
        self.classifier = MLPClassifier(**self.params)
        self.pipeline = Pipeline([
            ("scaler", self.scaler),
            ("classifier", self.classifier),
        ])
        self.model = self.pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MLPModel":
        self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        self.model = self.pipeline
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

