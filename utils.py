"""
utils.py — Shared utility functions.

Saves class names to a JSON file after training so the Streamlit
deployment app does not need access to the original dataset.
"""

import json
import config


def save_class_names(class_names: list) -> None:
    """Persist class names to disk alongside model checkpoints."""
    with open(config.CLASS_NAMES_PATH, "w") as fh:
        json.dump(class_names, fh, indent=2)
    print(f"[utils] Class names saved → {config.CLASS_NAMES_PATH}")


def load_class_names() -> list:
    """Load class names that were saved during training."""
    with open(config.CLASS_NAMES_PATH, "r") as fh:
        class_names = json.load(fh)
    print(f"[utils] Loaded {len(class_names)} class names.")
    return class_names
