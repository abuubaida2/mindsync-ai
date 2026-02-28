"""
Reproducibility seed utilities.
Sets seed for PyTorch, NumPy, and Python random (paper: seed=42).
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set global seeds for deterministic training.

    Sets seed for: Python random, NumPy, PyTorch (CPU + GPU),
    CUDA deterministic algorithms.

    Args:
        seed: Random seed. Paper uses seed=42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
