
"""Utilities shared across the classification training pipeline."""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    """Save a Python object as a formatted JSON file."""

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """Load and return a JSON file."""

    with open(path) as f:
        return json.load(f)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        """Update the stopping state using the current metric value."""

        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best
            if self.mode == "max"
            else value < self.best
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
