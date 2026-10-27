"""
Production XGBoost Classifier for Stack Overflow Accepted Answer Prediction.
Incorporates optimal hyperparameters (lr=0.1, max_depth=15, n_estimators=450),
cross-validation routines, and feature importance analysis.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from src.models.base import BaseModel
from src.config import DEFAULT_XGB_PARAMS
from src.utils import setup_logger

logger = setup_logger("xgboost_model")

class XGBoostModel(BaseModel):
    """
    Optimized Extreme Gradient Boosting Classifier.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        super().__init__("XGBoost", merged_params)
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[List] = None, verbose: bool = False, **kwargs) -> "XGBoostModel":
        self.feature_names = list(X.columns)
        fit_kwargs = {"verbose": verbose}
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
        """
        Executes Stratified K-Fold Cross-Validation across accuracy, precision, recall, and F1.
        """
        self.feature_names = list(X.columns)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        logger.info(f"Running {n_splits}-Fold Stratified Cross-Validation on {len(X):,} samples...")
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
            "raw_cv": cv_results,
        }

        # Calculate mean feature importances across all CV estimators
        all_importances = [est.feature_importances_ for est in cv_results["estimator"]]
        mean_importance = np.mean(all_importances, axis=0)
        summary["feature_importance"] = pd.Series(mean_importance, index=self.feature_names).sort_values(ascending=False)

        logger.info(f"CV Results -> Accuracy: {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f} | "
                    f"Precision: {summary['precision_mean']:.3f} | "
                    f"Recall: {summary['recall_mean']:.3f} | "
                    f"F1: {summary['f1_mean']:.3f} | "
                    f"ROC-AUC: {summary['roc_auc_mean']:.3f}")

        return summary

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 2,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        """
        Executes hyperparameter sweep matching notebook experimental setup.
        """
        grid = param_grid or {
            "learning_rate": [0.05, 0.1],
            "max_depth": [12, 15],
            "n_estimators": [350, 450],
        }
        logger.info(f"Initiating GridSearchCV across {grid} with {cv}-fold CV...")
        base_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X, y)
        logger.info(f"Optimal Hyperparameters: {grid_search.best_params_} (Best {scoring}: {grid_search.best_score_:.4f})")
        self.params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.feature_names = list(X.columns)
        return grid_search.best_params_
