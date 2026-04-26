"""
models/ensemble.py — Soft Voting Ensemble (heterogeneous fusion layer).

Fusion strategy
---------------
Unweighted soft voting: the softmax probability vectors from all three base
models are averaged element-wise.  The class with the highest average
probability is selected as the final diagnosis.

Mathematical formulation
------------------------
For N models and C classes:

    P_final[c] = (1/N) * Σ_{i=1}^{N} softmax(logits_i)[c]

    prediction = argmax_c  P_final[c]

Why soft voting over hard voting?
  Hard voting discards each model's confidence — a model barely crossing the
  decision boundary counts the same as one predicting with 99% certainty.
  Soft voting preserves this probabilistic information, leading to a more
  reliable and stable decision boundary (Wang, Chen, & Liu, 2024).

Why unweighted rather than weighted?
  Weighted averaging requires a held-out validation pass to determine weights,
  adding complexity and risk of overfitting to the validation set.  The
  unweighted average is theoretically sound when base models perform within a
  similar range, which is expected after proper fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVotingEnsemble(nn.Module):
    """
    Aggregates ResNet50, MobileNetV3, and ViT predictions via soft voting.

    Parameters
    ----------
    resnet    : nn.Module  — fine-tuned ResNet50
    mobilenet : nn.Module  — fine-tuned MobileNetV3-Large
    vit       : nn.Module  — fine-tuned ViT-Base/16 (HuggingFace)
    """

    def __init__(self, resnet: nn.Module, mobilenet: nn.Module, vit: nn.Module):
        super().__init__()
        self.resnet    = resnet
        self.mobilenet = mobilenet
        self.vit       = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, 224, 224)

        Returns
        -------
        avg_prob : torch.Tensor  shape (B, num_classes)
            Averaged softmax probabilities — the final ensemble output.
        """
        # ── ResNet50 ──────────────────────────────────────────────────────────
        logits_r = self.resnet(x)
        prob_r   = F.softmax(logits_r, dim=1)

        # ── MobileNetV3 ───────────────────────────────────────────────────────
        logits_m = self.mobilenet(x)
        prob_m   = F.softmax(logits_m, dim=1)

        # ── ViT (HuggingFace API returns a dataclass, not a raw tensor) ───────
        vit_output = self.vit(pixel_values=x)
        logits_v   = vit_output.logits
        prob_v     = F.softmax(logits_v, dim=1)

        # ── Unweighted average ────────────────────────────────────────────────
        avg_prob = (prob_r + prob_m + prob_v) / 3.0
        return avg_prob
