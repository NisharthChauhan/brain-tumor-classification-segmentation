
"""Training utilities for the BRISC2025 classification experiments."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from augmentations import (
    get_classification_train_transforms,
    get_classification_eval_transforms,
)
from brisc_datasets import get_classification_loaders
from models_classification import (
    get_classification_model,
    count_parameters,
)
from utils import set_seed, save_json, EarlyStopping


def train_classification_model(
    model_name,
    project_root,
    cls_split,
    seed=42,
    batch_size=32,
    lr=1e-4,
    max_epochs=50,
    patience=10,
    img_size=224,
    device=None,
):
    """Train and evaluate one classification model."""

    set_seed(seed)

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_dir = os.path.join(
        project_root,
        "results",
        "classification",
        model_name,
        f"seed_{seed}",
    )

    os.makedirs(run_dir, exist_ok=True)

    # Build the train and validation loaders using the shared pipeline.
    train_loader, val_loader, class_to_idx = get_classification_loaders(
        cls_split,
        None,
        batch_size=batch_size,
        train_transform=get_classification_train_transforms(img_size),
        eval_transform=get_classification_eval_transforms(img_size),
        num_workers=2,
    )

    num_classes = len(class_to_idx)

    # Build the selected model with ImageNet-pretrained weights.
    model = get_classification_model(
        model_name,
        num_classes=num_classes,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(
        patience=patience,
        mode="max",
    )

    history = []
    best_val_acc = -1.0
    best_epoch = -1

    best_ckpt_path = os.path.join(
        run_dir,
        "best_model.pt",
    )

    separator = "=" * 60

    print(f"\n{separator}")
    print(f"Training {model_name} (seed={seed}) on {device}")
    print(separator)

    print(
        f"Total params: {total_params / 1e6:.2f}M | "
        f"Trainable: {trainable_params / 1e6:.2f}M"
    )

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Training
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        })

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Save the checkpoint with the best validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                best_ckpt_path,
            )

        if early_stopper.step(val_acc):
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(best val_acc={best_val_acc:.4f} "
                f"at epoch {best_epoch + 1})"
            )
            break

    # Evaluate the best checkpoint on the validation set.
    model.load_state_dict(
        torch.load(
            best_ckpt_path,
            map_location=device,
        )
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_preds.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probs.extend(
                probabilities.cpu().numpy()
            )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )
    )

    try:
        macro_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = None

    summary = {
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_auc": (
            float(macro_auc)
            if macro_auc is not None
            else None
        ),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_epoch_time_sec": float(
            np.mean(
                [h["epoch_time_sec"] for h in history]
            )
        ),
        "num_epochs_trained": len(history),
    }

    # Save the training history and experiment summary.
    save_json(
        history,
        os.path.join(run_dir, "history.json"),
    )

    save_json(
        summary,
        os.path.join(run_dir, "summary.json"),
    )

    print(f"\n{separator}")

    print(
        f"Finished {model_name}: "
        f"best_val_acc={best_val_acc:.4f} "
        f"(epoch {best_epoch + 1}), "
        f"macro_f1={f1:.4f}"
    )

    print(separator)

    return summary
