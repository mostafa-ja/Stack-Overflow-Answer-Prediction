"""
Abstract Base Model Interface for all Stack Overflow Answer Prediction classifiers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib

class BaseModel(ABC):
    """Abstract base class ensuring a unified interface across all algorithms."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_names = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """Trains the underlying estimator on features X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates binary predictions (0 or 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predicted probability estimates for positive class (accepted)."""
        pass

    def get_feature_importances(self) -> Optional[pd.Series]:
        """Returns feature importances if supported by the model architecture."""
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return None

    def save(self, output_path: Path) -> Path:
        """Serializes trained model artifact to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "params": self.params, "features": self.feature_names, "name": self.name}, output_path)
        return output_path

    def load(self, model_path: Path) -> "BaseModel":
        """Deserializes model artifact from disk."""
        data = joblib.load(model_path)
        self.model = data["model"]
        self.params = data.get("params", {})
        self.feature_names = data.get("features", [])
        self.name = data.get("name", self.name)
        return self
