"""
train.py — Training loop for all three base models.

Run this script on Google Colab (GPU runtime recommended).
Models are saved to CHECKPOINT_DIR in config.py after each epoch where
validation accuracy improves.  Early stopping prevents wasted compute.

Usage (in Colab cell):
    %run train.py
or:
    !python train.py
"""

import copy
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import get_dataloaders
from models.resnet_model import build_resnet50
from models.mobilenet_model import build_mobilenetv3
from models.vit_model import build_vit
from utils import save_class_names

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[train] GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch(model, loader, criterion, optimizer, is_vit: bool) -> tuple:
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if is_vit:
            logits = model(pixel_values=images).logits
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients with ViT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds    = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, is_vit: bool) -> tuple:
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_vit:
            logits = model(pixel_values=images).logits
        else:
            logits = model(images)

        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds    = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model,
    model_name: str,
    checkpoint_path: str,
    train_loader,
    val_loader,
    num_epochs: int = config.NUM_EPOCHS,
    is_vit: bool = False,
) -> tuple:
    """
    Train a single model and return (trained_model, history_dict).

    Saves the best checkpoint (highest val accuracy) to checkpoint_path.
    Applies early stopping after config.PATIENCE epochs without improvement.
    """
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Only pass parameters that require gradients to the optimiser
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine annealing: LR decays smoothly from LEARNING_RATE → eta_min
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_acc  = 0.0
    best_weights  = None
    patience_ctr  = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    print(f"\n{'=' * 65}")
    print(f"  Training: {model_name}")
    print(f"{'=' * 65}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = _train_epoch(
            model, train_loader, criterion, optimizer, is_vit
        )
        vl_loss, vl_acc = _eval_epoch(
            model, val_loader, criterion, is_vit
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        status = "✓" if vl_acc > best_val_acc else " "
        print(
            f"  [{status}] Epoch {epoch:02d}/{num_epochs}  "
            f"Train  Loss: {tr_loss:.4f}  Acc: {tr_acc * 100:6.2f}%  |  "
            f"Val  Loss: {vl_loss:.4f}  Acc: {vl_acc * 100:6.2f}%  "
            f"({elapsed:.0f}s)"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, checkpoint_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {config.PATIENCE} epochs).")
                break

    # Restore the best weights before returning
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    print(
        f"\n  Best validation accuracy for {model_name}: "
        f"{best_val_acc * 100:.2f}%"
    )
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, num_classes, class_names = (
        get_dataloaders()
    )

    # Persist class names so the Streamlit app does not need the dataset
    save_class_names(class_names)

    # ── ResNet50 ──────────────────────────────────────────────────────────────
    resnet = build_resnet50(num_classes)
    resnet, resnet_hist = train_model(
        resnet, "ResNet50", config.RESNET_CKPT,
        train_loader, val_loader,
    )

    # ── MobileNetV3-Large ─────────────────────────────────────────────────────
    mobilenet = build_mobilenetv3(num_classes)
    mobilenet, mobilenet_hist = train_model(
        mobilenet, "MobileNetV3-Large", config.MOBILENET_CKPT,
        train_loader, val_loader,
    )

    # ── ViT-Base/16 (ImageNet-21k) ────────────────────────────────────────────
    vit = build_vit(num_classes)
    vit, vit_hist = train_model(
        vit, "ViT-Base/16 (IN-21k)", config.VIT_CKPT,
        train_loader, val_loader,
        is_vit=True,
    )

    print("\n[train] All three models trained successfully.")
    print("[train] Run  python evaluate.py  to generate test metrics and plots.")

    return resnet, mobilenet, vit, test_loader, num_classes, class_names


if __name__ == "__main__":
    main()
