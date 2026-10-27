"""
Production LightGBM Classifier for Stack Overflow Accepted Answer Prediction.
Provides fast leaf-wise gradient boosting alternative to XGBoost.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.models.base import BaseModel
from src.config import DEFAULT_LGBM_PARAMS
from src.utils import setup_logger

logger = setup_logger("lightgbm_model")

class LightGBMModel(BaseModel):
    """
    LightGBM Gradient Boosting Framework.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        super().__init__("LightGBM", merged_params)
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[List] = None, **kwargs) -> "LightGBMModel":
        self.feature_names = list(X.columns)
        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 10,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Runs Stratified K-Fold CV for LightGBM."""
        self.feature_names = list(X.columns)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        logger.info(f"Running {n_splits}-Fold Stratified CV on {len(X):,} samples (LightGBM)...")
        cv_results = cross_validate(self.model, X, y, cv=cv, scoring=scoring, return_estimator=True)

        summary = {
            "accuracy_mean": float(cv_results["test_accuracy"].mean()),
            "accuracy_std": float(cv_results["test_accuracy"].std()),
            "precision_mean": float(cv_results["test_precision"].mean()),
            "precision_std": float(cv_results["test_precision"].std()),
            "recall_mean": float(cv_results["test_recall"].mean()),
            "recall_std": float(cv_results["test_recall"].std()),
            "f1_mean": float(cv_results["test_f1"].mean()),
            "f1_std": float(cv_results["test_f1"].std()),
            "roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
            "roc_auc_std": float(cv_results["test_roc_auc"].std()),
        }

        all_importances = [est.feature_importances_ for est in cv_results["estimator"]]
        mean_importance = np.mean(all_importances, axis=0)
        summary["feature_importance"] = pd.Series(mean_importance, index=self.feature_names).sort_values(ascending=False)

        return summary
