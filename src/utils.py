"""
General utilities: logging, reproducibility seeds, hardware checks, and plotting styles.
"""

import os
import sys
import random
import logging
from typing import Dict, Any
from pathlib import Path
import numpy as np

def setup_logger(name: str = "so_prediction", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def seed_everything(seed: int = 42) -> None:
    """Ensures deterministic execution across Python, NumPy, and environment."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_system_info() -> Dict[str, Any]:
    """Inspects available hardware (CPU cores, RAM, GPU/CUDA)."""
    import platform
    import multiprocessing

    info: Dict[str, Any] = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_available": False,
        "gpu_name": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    return info

def ensure_directories(*dirs: Path) -> None:
    """Ensures that all specified directory paths exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def setup_plot_style() -> None:
    """Sets a polished, modern matplotlib/seaborn aesthetic."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "figure.autolayout": True,
            "figure.titlesize": 14,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "lines.linewidth": 2.0,
            "grid.alpha": 0.3,
        })
    except ImportError:
        pass
