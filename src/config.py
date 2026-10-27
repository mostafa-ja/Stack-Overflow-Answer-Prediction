"""
Configuration and constants for the Stack Overflow Answer Prediction pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_saved"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Remote Dataset Identifiers (Google Drive)
ACCEPTED_DF_GDRIVE_ID = "1oDop0IQkV9k0xwNT5wL2jp4PHBgLj13X"
RAW_DATA_GDRIVE_ID = "1quMzXrCUCVTHaVTAT26jO-T_HnhSBxtp"

# Target Definition
TARGET_COL = "accepted"

# Full Feature Set (Retrospective observation including post-hoc community reaction)
FEATURE_COLS_FULL: List[str] = [
    "Score_A",
    "CommentCount_A",
    "Score_Q",
    "ViewCount_Q",
    "CommentCount_Q",
    "FavoriteCount_Q",
    "Reputation",
    "Views_U",
    "UpVotes",
    "DownVotes",
    "badgeCount",
    "Closed_Q",
    "Time_Diffrence(H)",
    "Accepted_Ratio_A",
    "#answers_Q",
    "Earlier_Answers",
    "Accepted_Rratio_Q",
    "accepted_ratio_tags_mean",
    "#tags_Q",
]

# Real-Time Feature Set (Strict submission-time features, zero future leakage)
FEATURE_COLS_REALTIME: List[str] = [
    "Reputation",
    "Views_U",
    "UpVotes",
    "DownVotes",
    "badgeCount",
    "Time_Diffrence(H)",
    "Accepted_Ratio_A",
    "Earlier_Answers",
    "Accepted_Rratio_Q",
    "accepted_ratio_tags_mean",
    "#tags_Q",
]

# Leaked / Post-hoc features dropped for Real-Time setting
LEAKAGE_COLS: List[str] = [
    "Score_Q",
    "CommentCount_A",
    "Score_A",
    "#answers_Q",
    "Closed_Q",
    "ViewCount_Q",
    "CommentCount_Q",
    "FavoriteCount_Q",
]

# Optimal Hyperparameters discovered via 2-fold / 10-fold CV GridSearch
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.1,
    "max_depth": 15,
    "n_estimators": 450,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.1,
    "max_depth": 15,
    "num_leaves": 63,
    "n_estimators": 450,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 12,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    "max_iter": 1000,
    "random_state": 42,
}
