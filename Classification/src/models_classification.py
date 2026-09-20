
"""Model factory for the classification architectures used in this study.

Supported models:
    VGG19
    ResNet50
    DenseNet121
    EfficientNet-B0
    EfficientNet-B1
    EfficientNet-B4
"""

import timm


MODEL_REGISTRY = {
    "vgg19": "vgg19",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b4": "efficientnet_b4",
}

CLASSIFICATION_MODEL_NAMES = list(MODEL_REGISTRY.keys())


def get_classification_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_strategy: str = "none"
):
    """Build a classification model from the supported architectures.

    Args:
        name: Model name from CLASSIFICATION_MODEL_NAMES.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.
        freeze_strategy:
            "none"    - Keep all parameters trainable.
            "partial" - Freeze approximately the first 75% of parameters.
            "full"    - Freeze all parameters except the classifier head.

    Returns:
        A configured timm classification model.
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {CLASSIFICATION_MODEL_NAMES}"
        )

    model = timm.create_model(
        MODEL_REGISTRY[name],
        pretrained=pretrained,
        num_classes=num_classes
    )

    if freeze_strategy == "full":
        _freeze_all_but_head(model)

    elif freeze_strategy == "partial":
        _freeze_partial(model)

    elif freeze_strategy != "none":
        raise ValueError(
            f"Unknown freeze strategy '{freeze_strategy}'. "
            "Choose from: none, partial, full."
        )

    return model


def _freeze_all_but_head(model):
    """Freeze all parameters except the final classifier head."""

    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()

    for param in classifier.parameters():
        param.requires_grad = True


def _freeze_partial(model):
    """Freeze approximately 75% of parameters.

    The remaining parameters, including the later layers and
    classifier head, remain trainable.
    """

    all_params = list(model.named_parameters())
    cutoff = int(len(all_params) * 0.75)

    for i, (_, param) in enumerate(all_params):
        param.requires_grad = i >= cutoff


def count_parameters(model):
    """Return total and trainable parameter counts."""

    total_params = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    return total_params, trainable_params
