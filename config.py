"""
config.py — Central configuration for the Plant Pathology Diagnosis System.
All paths, hyperparameters, and constants live here.
Update DATA_ROOT and CHECKPOINT_DIR before running on Colab.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# Update these to match your Google Drive layout on Colab.
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT       = "/content/drive/MyDrive/PlantDiseaseDataset"
TRAIN_DIR       = os.path.join(DATA_ROOT, "train")
VALID_DIR       = os.path.join(DATA_ROOT, "valid")
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TARGET CROPS  (only Tomato, Potato, and Pepper are used in this study)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_PREFIXES = ("Tomato", "Potato", "Pepper")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SETTINGS
# 224 × 224 is compatible with ResNet50, MobileNetV3-Large, and ViT-Base/16.
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
BATCH_SIZE   = 32      # Works well on Colab T4 (16 GB VRAM)
NUM_WORKERS  = 2

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-4     # AdamW base learning rate
WEIGHT_DECAY    = 1e-2     # L2 regularisation
NUM_EPOCHS      = 20       # Maximum epochs per model
PATIENCE        = 5        # Early-stopping patience (epochs without improvement)
LABEL_SMOOTHING = 0.1      # Reduces overconfidence during training

# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT STRATEGY
# The dataset's existing train/ folder → training.
# The dataset's existing valid/ folder is split 50/50 → validation / test.
# This yields approximately the 70 / 15 / 15 split stated in the methodology.
# ─────────────────────────────────────────────────────────────────────────────
VAL_TEST_SPLIT = 0.5   # Fraction of valid/ reserved for testing

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT PATHS
# ─────────────────────────────────────────────────────────────────────────────
RESNET_CKPT       = os.path.join(CHECKPOINT_DIR, "resnet50_best.pth")
MOBILENET_CKPT    = os.path.join(CHECKPOINT_DIR, "mobilenetv3_best.pth")
VIT_CKPT          = os.path.join(CHECKPOINT_DIR, "vit_best.pth")
CLASS_NAMES_PATH  = os.path.join(CHECKPOINT_DIR, "class_names.json")

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM TARGET LAYERS
# These are the final convolutional blocks of each CNN.
# Grad-CAM hooks are registered on these named modules.
# ─────────────────────────────────────────────────────────────────────────────
RESNET_TARGET_LAYER    = "layer4"        # Last residual block in ResNet50
MOBILENET_TARGET_LAYER = "features.16"  # Last conv block in MobileNetV3-Large
