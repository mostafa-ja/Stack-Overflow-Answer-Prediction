"""
Comprehensive evaluation, metric calculation, error analysis, and visualization utilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.config import FIGURES_DIR, REPORTS_DIR
from src.utils import setup_logger, setup_plot_style, ensure_directories

logger = setup_logger("evaluate")

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    return metrics

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generates a styled confusion matrix heatmap."""
    setup_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Accepted", "Accepted"],
        yticklabels=["Not Accepted", "Accepted"],
        ax=ax,
    )
    # Overlay percentages
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="gray" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=9,
            )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_directories(save_path.parent)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved confusion matrix plot to {save_path}")

    return fig

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots ROC curve with AUC score."""
    setup_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Random Guess")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"Receiver Operating Characteristic (ROC) - {model_name}", pad=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_directories(save_path.parent)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved ROC curve plot to {save_path}")

    return fig

def plot_feature_importance(
    importance_series: pd.Series,
    title: str = "Mean Feature Importance",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots horizontal bar chart of top features."""
    setup_plot_style()
    top_features = importance_series.head(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(top_features))
    ax.barh(top_features.index, top_features.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Relative Importance Score")
    ax.set_title(title, pad=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_directories(save_path.parent)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved feature importance plot to {save_path}")

    return fig

def perform_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Performs stratified error analysis diagnosing False Positives and False Negatives.
    """
    analysis_df = X_test.copy()
    analysis_df["true_label"] = y_test.values
    analysis_df["predicted_label"] = y_pred
    analysis_df["predicted_prob"] = y_prob

    # Classify error types
    analysis_df["error_type"] = "Correct"
    analysis_df.loc[(analysis_df["true_label"] == 0) & (analysis_df["predicted_label"] == 1), "error_type"] = "False Positive"
    analysis_df.loc[(analysis_df["true_label"] == 1) & (analysis_df["predicted_label"] == 0), "error_type"] = "False Negative"

    fp_count = (analysis_df["error_type"] == "False Positive").sum()
    fn_count = (analysis_df["error_type"] == "False Negative").sum()

    report: Dict[str, Any] = {
        "false_positive_count": int(fp_count),
        "false_negative_count": int(fn_count),
        "insights": {},
    }

    # Inspect feature means by error type
    numeric_cols = [c for c in ["Time_Diffrence(H)", "Earlier_Answers", "Reputation", "Accepted_Ratio_A"] if c in analysis_df.columns]
    if numeric_cols:
        grouped_means = analysis_df.groupby("error_type")[numeric_cols].mean().to_dict()
        report["feature_means_by_error"] = grouped_means

    return report
