
"""PyTorch dataset and DataLoader utilities for BRISC2025 classification."""

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class BRISCClassificationDataset(Dataset):
    """Load image and class-label pairs from the saved split."""

    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, classname = self.samples[idx]

        image = Image.open(filepath).convert("RGB")
        image = np.array(image)

        label = self.class_to_idx[classname]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


def get_classification_loaders(
    split_json,
    images_root_unused,
    batch_size,
    train_transform,
    eval_transform,
    num_workers=2
):
    """Create train and validation DataLoaders from the saved split."""

    classes = split_json["classes"]
    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    train_dataset = BRISCClassificationDataset(
        split_json["train"],
        class_to_idx,
        train_transform
    )

    val_dataset = BRISCClassificationDataset(
        split_json["val"],
        class_to_idx,
        eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_to_idx
