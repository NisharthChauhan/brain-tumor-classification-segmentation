
"""Utilities for creating reproducible classification data splits."""

import glob
import json
import os
import random


def make_classification_split(
    cls_train_dir,
    out_path,
    val_fraction=0.2,
    seed=42
):
    """Create and save a reproducible train/validation split.

    The BRISC2025 classification training set is split within each
    class so that the class distribution is preserved.

    Args:
        cls_train_dir: Directory containing one folder per class.
        out_path: Path where the split JSON file will be saved.
        val_fraction: Fraction of each class used for validation.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary containing the class names and split samples.
    """

    random.seed(seed)

    classes = sorted(os.listdir(cls_train_dir))

    train_files = []
    val_files = []

    for class_name in classes:
        class_dir = os.path.join(cls_train_dir, class_name)

        files = sorted(
            glob.glob(os.path.join(class_dir, "*.jpg"))
        )

        random.shuffle(files)

        n_val = int(len(files) * val_fraction)

        val_files.extend(
            [(filepath, class_name) for filepath in files[:n_val]]
        )

        train_files.extend(
            [(filepath, class_name) for filepath in files[n_val:]]
        )

    split = {
        "classes": classes,
        "train": train_files,
        "val": val_files,
    }

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(
        f"Classification split saved: "
        f"{len(train_files)} train, {len(val_files)} val"
    )

    return split
