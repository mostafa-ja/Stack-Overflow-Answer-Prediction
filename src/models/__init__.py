"""
Model registry and factory for Stack Overflow Answer Prediction.
"""

from typing import Optional, Dict, Any
from src.models.base import BaseModel
from src.models.baseline import LogisticRegressionModel, RandomForestModel, MLPModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "randomforest": RandomForestModel,
    "logisticregression": LogisticRegressionModel,
    "mlp": MLPModel,
}

def get_model(name: str, params: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Factory function returning an instantiated model matching registry name."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](params=params)

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "MLPModel",
    "XGBoostModel",
    "LightGBMModel",
    "get_model",
    "MODEL_REGISTRY",
]
