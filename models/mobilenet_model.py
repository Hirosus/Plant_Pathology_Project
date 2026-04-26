"""
models/mobilenet_model.py — MobileNetV3-Large branch of the ensemble.

Architecture choice
-------------------
MobileNetV3-Large pretrained on ImageNet-1k (V2 weights).
The bulk of the feature extractor (features[0:14]) is frozen to preserve
generalised representations. The last three convolutional blocks
(features[14:17]) are unfrozen and fine-tuned alongside the classifier.

Why MobileNetV3?
  - Depth-wise separable convolutions drastically reduce parameter count
    without sacrificing meaningful diagnostic accuracy.
  - Its efficient architecture complements the heavier ResNet50: the ensemble
    gains fast, efficient local feature extraction from MobileNetV3.
  - 'features.16' (the last Conv2dNormActivation block) is an ideal Grad-CAM
    target: it outputs spatially detailed feature maps before global pooling.
  - Suitable for the eventual lightweight local inference phase on the
    Dell Precision 3550 (Singh & Sharma, 2024).
"""

import torch.nn as nn
from torchvision import models


def build_mobilenetv3(num_classes: int) -> nn.Module:
    """
    Return a MobileNetV3-Large fine-tuned for plant disease classification.

    Parameters
    ----------
    num_classes : int
        Number of target disease / healthy classes.

    Returns
    -------
    nn.Module
        MobileNetV3-Large with a custom classifier head.
    """
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    # ── Freeze early feature blocks ───────────────────────────────────────────
    for param in model.features[:14].parameters():
        param.requires_grad = False

    # ── Unfreeze last 3 feature blocks for domain-specific fine-tuning ────────
    for param in model.features[14:].parameters():
        param.requires_grad = True

    # ── Replace the classifier head ───────────────────────────────────────────
    # Original MobileNetV3-Large classifier: [Linear(960, 1280), HardSwish,
    #                                         Dropout, Linear(1280, 1000)]
    in_features = model.classifier[3].in_features   # 1280
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[MobileNetV3] Trainable params: {trainable:,} / {total:,}")
    return model
