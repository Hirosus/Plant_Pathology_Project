"""
evaluate.py — Full evaluation of all three base models and the ensemble.

Produces:
  - Per-model and ensemble: Accuracy, Precision, Recall, F1 (weighted)
  - Per-class classification report
  - Confusion matrix plots (saved as PNG)
  - Bar chart comparison across all four systems

Run after train.py has completed:
    python evaluate.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from dataset import get_dataloaders
from models.ensemble import SoftVotingEnsemble
from models.mobilenet_model import build_mobilenetv3
from models.resnet_model import build_resnet50
from models.vit_model import build_vit
from utils import load_class_names

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = config.CHECKPOINT_DIR   # Save plots alongside checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_all_models(num_classes: int) -> tuple:
    resnet = build_resnet50(num_classes)
    resnet.load_state_dict(
        torch.load(config.RESNET_CKPT, map_location=DEVICE)
    )
    resnet = resnet.to(DEVICE).eval()

    mobilenet = build_mobilenetv3(num_classes)
    mobilenet.load_state_dict(
        torch.load(config.MOBILENET_CKPT, map_location=DEVICE)
    )
    mobilenet = mobilenet.to(DEVICE).eval()

    vit = build_vit(num_classes)
    vit.load_state_dict(
        torch.load(config.VIT_CKPT, map_location=DEVICE)
    )
    vit = vit.to(DEVICE).eval()

    return resnet, mobilenet, vit


# ─────────────────────────────────────────────────────────────────────────────
# Prediction collection
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, is_vit: bool = False) -> tuple:
    """Return (predictions, true_labels, probability_matrix)."""
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)

        if is_vit:
            logits = model(pixel_values=images).logits
        else:
            logits = model(images)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


@torch.no_grad()
def collect_ensemble_predictions(ensemble, loader) -> tuple:
    """Return (predictions, true_labels, probability_matrix) for the ensemble."""
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images   = images.to(DEVICE, non_blocking=True)
        avg_prob = ensemble(images)
        preds    = avg_prob.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(avg_prob.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric reporting
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_print_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  Accuracy   : {acc  * 100:.2f}%")
    print(f"  Precision  : {prec * 100:.2f}%")
    print(f"  Recall     : {rec  * 100:.2f}%")
    print(f"  F1 Score   : {f1   * 100:.2f}%")
    print(f"\n  Per-Class Report:\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str,
    filename: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n  = len(class_names)
    fig_size = max(10, n)   # Scale figure with number of classes

    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,             fontsize=7)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_metric_comparison(results: dict, metric: str) -> None:
    """Bar chart comparing all models on a given metric."""
    names  = list(results.keys())
    values = [results[n][metric] * 100 for n in names]

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        names, values,
        color=colours[: len(names)],
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel(f"{metric.capitalize()} (%)", fontsize=12)
    ax.set_title(
        f"Model Comparison — {metric.capitalize()}",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(y=90, color="red", linestyle="--", linewidth=1,
               label="90% target line")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_training_curves(history: dict, model_name: str) -> None:
    """Loss and accuracy curves for a single model's training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, label="Train Acc", marker="o")
    axes[1].plot(epochs, val_acc_pct,   label="Val Acc",   marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{model_name} — Accuracy Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(SAVE_DIR, f"curves_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    class_names = load_class_names()
    num_classes = len(class_names)

    _, _, test_loader, _, _ = get_dataloaders()

    resnet, mobilenet, vit = load_all_models(num_classes)

    # Build ensemble
    ensemble = SoftVotingEnsemble(resnet, mobilenet, vit).to(DEVICE)
    ensemble.eval()

    all_results = {}

    # ── Individual model evaluation ───────────────────────────────────────────
    configs = [
        (resnet,    "ResNet50",         "cm_resnet50.png",    False),
        (mobilenet, "MobileNetV3-Large","cm_mobilenetv3.png", False),
        (vit,       "ViT-Base/16",      "cm_vit.png",         True),
    ]

    for model, name, cm_file, is_vit in configs:
        preds, labels, _ = collect_predictions(model, test_loader, is_vit)
        metrics = compute_and_print_metrics(name, labels, preds, class_names)
        all_results[name] = metrics
        plot_confusion_matrix(
            labels, preds, class_names,
            f"Confusion Matrix — {name}", cm_file,
        )

    # ── Ensemble evaluation ───────────────────────────────────────────────────
    ens_preds, ens_labels, _ = collect_ensemble_predictions(
        ensemble, test_loader
    )
    ens_metrics = compute_and_print_metrics(
        "Soft Voting Ensemble", ens_labels, ens_preds, class_names
    )
    all_results["Ensemble"] = ens_metrics
    plot_confusion_matrix(
        ens_labels, ens_preds, class_names,
        "Confusion Matrix — Soft Voting Ensemble",
        "cm_ensemble.png",
    )

    # ── Comparison charts ─────────────────────────────────────────────────────
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_metric_comparison(all_results, metric)

    print("\n[evaluate] All metrics computed and plots saved.")
    print(f"[evaluate] Results directory: {SAVE_DIR}")

    return all_results


if __name__ == "__main__":
    main()
