"""
dataset.py — Data loading, preprocessing, augmentation, and DataLoader creation.

Strategy
--------
* Load both train/ and valid/ folders using torchvision.datasets.ImageFolder.
* Filter to only the target crop prefixes (Tomato, Potato, Pepper).
* Remap class indices to a contiguous range [0, num_classes - 1].
* Split valid/ 50 / 50 into validation and test sets (stratified).
* Apply heavy augmentation to the training set; only resize + normalise elsewhere.
"""

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

import config


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_target_indices(dataset: datasets.ImageFolder,
                        prefixes: tuple) -> tuple:
    """
    Return:
      - indices  : list of sample indices whose class starts with any prefix
      - cls_idxs : list of original class indices that matched
    """
    cls_idxs = [
        i for i, cls_name in enumerate(dataset.classes)
        if cls_name.startswith(prefixes)
    ]
    indices = [
        i for i, (_, label) in enumerate(dataset.samples)
        if label in cls_idxs
    ]
    return indices, cls_idxs


def _build_label_map(original_classes: list,
                     target_cls_idxs: list) -> tuple:
    """
    Map original (sparse) class indices to new sequential indices.

    Returns:
      label_map   : dict {original_idx: new_idx}
      class_names : list of class name strings in new-index order
    """
    label_map   = {orig: new for new, orig in enumerate(target_cls_idxs)}
    class_names = [original_classes[i] for i in target_cls_idxs]
    return label_map, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Custom Dataset wrapper  (applies transform + label remapping on the fly)
# ─────────────────────────────────────────────────────────────────────────────
class _SubsetWithTransform(Dataset):
    """Wraps an ImageFolder, applies a transform, and remaps labels."""

    def __init__(self,
                 base_dataset: datasets.ImageFolder,
                 indices: list,
                 transform: transforms.Compose,
                 label_map: dict):
        self.base      = base_dataset
        self.indices   = indices
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, original_label = self.base.samples[self.indices[idx]]
        image = self.base.loader(path)
        image = self.transform(image)
        return image, self.label_map[original_label]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms() -> tuple:
    """
    Training transform: heavy augmentation to simulate real-world field variability.
    Validation / test transform: only resize + normalise (no stochastic ops).
    ImageNet mean and std are used since all three models were pretrained on ImageNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, eval_transform


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders() -> tuple:
    """
    Build and return (train_loader, val_loader, test_loader, num_classes, class_names).

    Dataset split
    -------------
    train/  → training set  (as provided by the New Plant Diseases Dataset)
    valid/  → split 50 / 50 into validation and test sets (stratified by class)
    """
    set_seed()
    train_tf, eval_tf = get_transforms()

    # ── Load raw datasets (no transform — we apply transforms in the wrapper) ─
    raw_train = datasets.ImageFolder(config.TRAIN_DIR)
    raw_valid = datasets.ImageFolder(config.VALID_DIR)

    # ── Filter to target crops ────────────────────────────────────────────────
    train_indices, target_cls_idxs = _get_target_indices(
        raw_train, config.TARGET_PREFIXES
    )
    valid_indices, _ = _get_target_indices(
        raw_valid, config.TARGET_PREFIXES
    )

    # ── Build a consistent label mapping from both folders ────────────────────
    label_map_train, class_names = _build_label_map(
        raw_train.classes, target_cls_idxs
    )

    # For valid/, find matching class indices (folder names are the same)
    valid_cls_idxs = [
        i for i, cls_name in enumerate(raw_valid.classes)
        if cls_name.startswith(config.TARGET_PREFIXES)
    ]
    label_map_valid = {
        orig: class_names.index(raw_valid.classes[orig])
        for orig in valid_cls_idxs
    }

    # ── Stratified split of valid/ → val and test ─────────────────────────────
    valid_labels = [raw_valid.samples[i][1] for i in valid_indices]
    val_indices, test_indices = train_test_split(
        valid_indices,
        test_size=config.VAL_TEST_SPLIT,
        random_state=config.SEED,
        stratify=valid_labels,
    )

    # ── Build Dataset objects ─────────────────────────────────────────────────
    train_ds = _SubsetWithTransform(raw_train, train_indices, train_tf, label_map_train)
    val_ds   = _SubsetWithTransform(raw_valid, val_indices,   eval_tf,  label_map_valid)
    test_ds  = _SubsetWithTransform(raw_valid, test_indices,  eval_tf,  label_map_valid)

    num_classes = len(class_names)

    print(f"[dataset] Number of classes  : {num_classes}")
    print(f"[dataset] Training samples   : {len(train_ds)}")
    print(f"[dataset] Validation samples : {len(val_ds)}")
    print(f"[dataset] Test samples       : {len(test_ds)}")
    print(f"[dataset] Classes:\n  " + "\n  ".join(
        f"{i:02d}. {n}" for i, n in enumerate(class_names)
    ))

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    common_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, num_classes, class_names
