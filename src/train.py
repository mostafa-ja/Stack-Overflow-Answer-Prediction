"""
Unified Command-Line Interface (CLI) for training and evaluating Stack Overflow Answer Prediction models.
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)
from src.utils import (
    setup_logger,
    seed_everything,
    get_system_info,
    ensure_directories,
)
from src.data_loader import load_dataset, get_train_val_test_splits
from src.models import get_model
from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    perform_error_analysis,
)

logger = setup_logger("train_cli")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Stack Overflow Accepted Answer Prediction models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "randomforest", "logisticregression", "mlp"],
        help="Algorithm to train.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="realtime",
        choices=["realtime", "full"],
        help="Feature configuration: 'realtime' (submission time, no leakage) or 'full' (retrospective).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to Accepted_df.csv. If None, attempts download or fallback.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Subsample N records for rapid prototyping/benchmarking.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If > 1, runs Stratified K-Fold Cross Validation. If 0, uses train/val/test split.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter grid search before final training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient boosting algorithms.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        help="Maximum tree depth for gradient boosting/tree algorithms.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=450,
        help="Number of boosting rounds/trees.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for full reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory to store model artifacts and figures.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    ensure_directories(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)
    out_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    ensure_directories(out_dir)

    logger.info("=" * 65)
    logger.info("Stack Overflow Accepted Answer Prediction Platform")
    logger.info("=" * 65)

    sys_info = get_system_info()
    logger.info(f"System: {sys_info['os']} | Python: {sys_info['python_version']} | CPUs: {sys_info['cpu_count']}")
    if sys_info.get("gpu_available"):
        logger.info(f"Hardware Acceleration: GPU ({sys_info['gpu_name']}) active.")

    # 1. Load Data
    X, y = load_dataset(
        data_path=args.data_path,
        feature_set=args.feature_set,
        n_samples=args.n_samples,
        allow_synthetic_fallback=True,
    )

    # Model specific parameter overrides
    custom_params = {}
    if args.model in ["xgboost", "lightgbm"]:
        custom_params.update({
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "random_state": args.seed,
        })
    elif args.model == "randomforest":
        custom_params.update({
            "max_depth": args.max_depth,
            "n_estimators": min(args.n_estimators, 200),
            "random_state": args.seed,
        })

    model = get_model(args.model, params=custom_params)

    # 2. Cross-Validation or Train/Test Split
    if args.cv_folds > 1 and hasattr(model, "cross_validate"):
        logger.info(f"Initiating {args.cv_folds}-Fold Cross Validation for {args.model.upper()}...")
        cv_summary = model.cross_validate(X, y, n_splits=args.cv_folds, random_state=args.seed)

        # Plot feature importance
        if "feature_importance" in cv_summary:
            fi_path = FIGURES_DIR / f"{args.model}_{args.feature_set}_cv_feature_importance.png"
            plot_feature_importance(cv_summary["feature_importance"], title=f"{args.model.upper()} Mean Feature Importance (CV)", save_path=fi_path)

        # Save metrics
        metrics_save_path = REPORTS_DIR / f"{args.model}_{args.feature_set}_cv_metrics.json"
        clean_cv = {k: v for k, v in cv_summary.items() if k not in ["raw_cv", "feature_importance"]}
        with open(metrics_save_path, "w") as f:
            json.dump(clean_cv, f, indent=2)
        logger.info(f"Saved cross-validation metrics to {metrics_save_path}")

    # Standard Train/Val/Test flow
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(X, y, random_state=args.seed)

    if args.tune and hasattr(model, "tune_hyperparameters"):
        logger.info("Executing Hyperparameter Grid Search...")
        model.tune_hyperparameters(X_train, y_train)

    logger.info(f"Training {model.name} on {len(X_train):,} training instances...")
    eval_set = [(X_val, y_val)] if args.model in ["xgboost", "lightgbm"] else None
    model.fit(X_train, y_train, eval_set=eval_set)

    # 3. Evaluation on Held-Out Test Set
    logger.info("Evaluating on held-out test partition...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    test_metrics = compute_all_metrics(y_test.values, y_pred, y_prob)
    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    if test_metrics.get("roc_auc"):
        logger.info(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    # 4. Generate Visualizations
    cm_path = FIGURES_DIR / f"{args.model}_{args.feature_set}_confusion_matrix.png"
    plot_confusion_matrix(y_test.values, y_pred, title=f"Confusion Matrix ({model.name} - {args.feature_set.capitalize()})", save_path=cm_path)

    if y_prob is not None:
        roc_path = FIGURES_DIR / f"{args.model}_{args.feature_set}_roc_curve.png"
        plot_roc_curve(y_test.values, y_prob, model_name=f"{model.name} ({args.feature_set.capitalize()})", save_path=roc_path)

    importances = model.get_feature_importances()
    if importances is not None:
        fi_path = FIGURES_DIR / f"{args.model}_{args.feature_set}_feature_importance.png"
        plot_feature_importance(importances, title=f"{model.name} Feature Importance", save_path=fi_path)

    # 5. Stratified Error Analysis
    error_diag = perform_error_analysis(X_test, y_test, y_pred, y_prob)

    # 6. Save Artifacts
    saved_model_path = out_dir / f"{args.model}_{args.feature_set}_model.joblib"
    model.save(saved_model_path)
    logger.info(f"Serialized trained model artifact to {saved_model_path}")

    report_payload = {
        "model": args.model,
        "feature_set": args.feature_set,
        "n_samples": len(X),
        "test_metrics": test_metrics,
        "error_analysis": error_diag,
        "parameters": model.params,
    }
    report_path = REPORTS_DIR / f"{args.model}_{args.feature_set}_report.json"
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Evaluation report written to {report_path}")
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
