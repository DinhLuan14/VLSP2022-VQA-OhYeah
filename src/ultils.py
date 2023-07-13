# src/utils.py

import torch
import random
import numpy as np


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_accuracy(preds, labels):
    """Compute accuracy score for predictions."""
    return (preds == labels).mean()
