"""
models/resnet_model.py — ResNet50 branch of the heterogeneous ensemble.

Architecture choice
-------------------
ResNet50 pretrained on ImageNet-1k (V2 weights — higher accuracy than V1).
The first two residual blocks (layer1, layer2) are frozen because their
low-level edge / colour detectors transfer well from ImageNet.
The later blocks (layer3, layer4) and the new classifier head are fully
trainable, enabling the network to specialise in pathological textures.

Why ResNet50?
  - Skip connections prevent vanishing gradients → enables deep feature learning.
  - 'layer4' (the final residual block) outputs spatially rich 7×7 feature maps
    at 2048 channels — ideal for Grad-CAM heatmap generation.
  - Well-documented performance on agricultural image tasks (Too et al., 2019).
"""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int) -> nn.Module:
    """
    Return a ResNet50 fine-tuned for plant disease classification.

    Parameters
    ----------
    num_classes : int
        Number of target disease / healthy classes.

    Returns
    -------
    nn.Module
        ResNet50 with a custom Dropout → Linear classifier head.
    """
    # Load ImageNet-1k V2 pretrained weights (best available for ResNet50)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # ── Freeze early layers (generic low-level features) ──────────────────────
    for name, param in model.named_parameters():
        if name.startswith(("layer1", "layer2", "conv1", "bn1")):
            param.requires_grad = False

    # ── Replace the fully connected classifier head ───────────────────────────
    in_features = model.fc.in_features   # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[ResNet50] Trainable params: {trainable:,} / {total:,}")
    return model
