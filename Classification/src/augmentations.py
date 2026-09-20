
"""Image preprocessing and augmentation for classification."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classification_train_transforms(img_size=224):
    """Return the training augmentation pipeline."""

    return A.Compose([
        A.Resize(img_size, img_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),

        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.3
        ),

        A.GaussNoise(
            std_range=(0.04, 0.2),
            p=0.2
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])


def get_classification_eval_transforms(img_size=224):
    """Return the evaluation pipeline without augmentation."""

    return A.Compose([
        A.Resize(img_size, img_size),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])
