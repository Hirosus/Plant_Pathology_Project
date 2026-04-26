"""
models/vit_model.py — Vision Transformer branch of the ensemble.

Architecture choice
-------------------
google/vit-base-patch16-224-in21k from HuggingFace.

Key reason for choosing the IN-21k variant over IN-1k:
  ImageNet-21k pretraining exposes the model to 14 million images across
  21,841 classes, producing representations far richer than the standard
  1k-class ImageNet.  This dramatically improves fine-tuning performance on
  domain-specific tasks like plant pathology, where subtle visual differences
  between disease classes demand nuanced feature discrimination.

The first 9 of 12 transformer encoder blocks are frozen.  Only the last 3
blocks and the classification head are trained.  This prevents catastrophic
forgetting of the powerful 21k representations while still allowing
domain adaptation.

Why ViT?
  - Self-attention captures long-range spatial dependencies across the entire
    leaf surface — e.g. linking a lesion at the leaf tip to discolouration at
    the stem — which CNN receptive fields inherently miss (Dosovitskiy et al., 2020).
  - Complements the local texture focus of ResNet50 and MobileNetV3, ensuring
    ensemble diversity of error.
"""

import torch.nn as nn
from transformers import ViTForImageClassification


def build_vit(num_classes: int) -> nn.Module:
    """
    Return a ViT-Base/16 (ImageNet-21k) fine-tuned for plant disease classification.

    Parameters
    ----------
    num_classes : int
        Number of target disease / healthy classes.

    Returns
    -------
    nn.Module  (ViTForImageClassification)
        The HuggingFace model with a replaced classification head.
        Input convention: keyword argument pixel_values=tensor.
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # Replace the original head silently
        attn_implementation="eager",    # Required to extract attention maps
    )

    # ── Freeze the first 9 encoder blocks (out of 12) ─────────────────────────
    for i, block in enumerate(model.vit.encoder.layer):
        if i < 9:
            for param in block.parameters():
                param.requires_grad = False

    # The classifier head (model.classifier) is always trainable by default.

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[ViT-Base/16-IN21k] Trainable params: {trainable:,} / {total:,}")
    return model
