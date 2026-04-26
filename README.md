# Plant Pathology Diagnosis System — Setup Guide

**Project:** Development of an Ensemble Network for Explainable Plant Pathology Diagnosis  
**Stack:** PyTorch · HuggingFace Transformers · Streamlit · Grad-CAM

---

## File Structure

```
plant_diagnosis/
├── config.py           ← All paths and hyperparameters (edit this first)
├── dataset.py          ← Data loading and augmentation pipeline
├── models/
│   ├── resnet_model.py ← ResNet50 branch
│   ├── mobilenet_model.py ← MobileNetV3-Large branch
│   ├── vit_model.py    ← ViT-Base/16 (ImageNet-21k) branch
│   └── ensemble.py     ← Soft voting fusion layer
├── utils.py            ← Save/load class names between training and app
├── train.py            ← Full training loop (run this on Colab)
├── evaluate.py         ← Test metrics, confusion matrices, comparison charts
├── gradcam.py          ← Grad-CAM + ViT attention map generation
├── app.py              ← Streamlit deployment interface
└── requirements.txt
```

---

## Step 1 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Download the dataset ZIP and upload it to your **Google Drive**.
3. Extract it so you have this structure on your Drive:

```
MyDrive/
└── PlantDiseaseDataset/
    ├── train/
    │   ├── Tomato___Early_blight/
    │   ├── Tomato___Late_blight/
    │   ├── Potato___Early_blight/
    │   ├── Pepper,_bell___Bacterial_spot/
    │   └── ... (all 38 class folders)
    └── valid/
        ├── Tomato___Early_blight/
        └── ...
```

---

## Step 2 — Upload Project Files to Colab

Upload the entire `plant_diagnosis/` folder to your Google Drive, for example:

```
MyDrive/
└── plant_diagnosis/
    ├── config.py
    ├── train.py
    └── ...
```

---

## Step 3 — Colab Setup

Open a new Colab notebook. In the first cell:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change to project directory
import os
os.chdir('/content/drive/MyDrive/plant_diagnosis')

# Install dependencies
!pip install -r requirements.txt -q
```

Verify GPU is available:

```python
import torch
print(torch.cuda.is_available())        # Should print True
print(torch.cuda.get_device_name(0))   # Should show T4 or A100
```

---

## Step 4 — Update config.py

Open `config.py` and confirm these two paths match your Drive layout:

```python
DATA_ROOT      = "/content/drive/MyDrive/PlantDiseaseDataset"
CHECKPOINT_DIR = "/content/drive/MyDrive/PlantCheckpoints"
```

The checkpoint directory is created automatically if it does not exist.

---

## Step 5 — Train All Three Models

```python
# In a Colab cell:
%run train.py
```

**Expected training time on Colab T4 GPU:**

| Model | Approx. Time (20 epochs) |
|---|---|
| ResNet50 | 35–50 minutes |
| MobileNetV3-Large | 25–35 minutes |
| ViT-Base/16 (IN-21k) | 50–70 minutes |
| **Total** | **~2–3 hours** |

**Tips to avoid Colab session timeouts:**
- Enable `Runtime → Change runtime type → GPU` before starting.
- The script saves checkpoints to Google Drive after every improving epoch,
  so if the session dies you do not lose progress.
- If a session times out mid-training, re-run `train.py` — it will retrain
  from scratch (checkpoints on Drive preserve the best weights already saved).

---

## Step 6 — Evaluate on the Test Set

```python
%run evaluate.py
```

This produces:
- Accuracy, Precision, Recall, F1 for all 3 base models + the ensemble
- Confusion matrix PNG for each model (saved to CHECKPOINT_DIR)
- Bar chart comparisons for all four metrics

---

## Step 7 — Generate Grad-CAM Explanations

```python
from gradcam import visualise_explanations
from models.resnet_model import build_resnet50
from models.mobilenet_model import build_mobilenetv3
from models.vit_model import build_vit
from utils import load_class_names
import torch, config

DEVICE = torch.device("cuda")
class_names = load_class_names()
num_classes  = len(class_names)

resnet    = build_resnet50(num_classes)
mobilenet = build_mobilenetv3(num_classes)
vit       = build_vit(num_classes)

resnet.load_state_dict(torch.load(config.RESNET_CKPT,    map_location=DEVICE))
mobilenet.load_state_dict(torch.load(config.MOBILENET_CKPT, map_location=DEVICE))
vit.load_state_dict(torch.load(config.VIT_CKPT,          map_location=DEVICE))

resnet = resnet.to(DEVICE).eval()
mobilenet = mobilenet.to(DEVICE).eval()
vit = vit.to(DEVICE).eval()

# Replace with any test image path
visualise_explanations(
    image_path  = "/path/to/test/leaf.jpg",
    resnet      = resnet,
    mobilenet   = mobilenet,
    vit         = vit,
    class_names = class_names,
    true_label  = "Tomato___Early_blight",   # optional
)
```

---

## Step 8 — Run the Streamlit App

### Option A — Run locally (after training is complete)

Copy the `plant_diagnosis/` folder and all checkpoint `.pth` files to your local machine, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Option B — Run on Colab via ngrok (for demo purposes)

```python
!pip install pyngrok -q
from pyngrok import ngrok

# Kill any existing Streamlit processes
!kill $(lsof -t -i:8501) 2>/dev/null; echo "cleared"

# Start Streamlit in the background
!streamlit run app.py --server.port 8501 &

import time; time.sleep(3)

# Open a public tunnel
tunnel = ngrok.connect(8501)
print("Streamlit URL:", tunnel.public_url)
```

Open the printed URL in your browser. The full diagnostic interface will be available.

---

## Expected Results

Based on literature and the architecture choices made in this study, expected
test-set performance on the filtered Tomato/Potato/Pepper subset:

| Model | Expected Accuracy |
|---|---|
| ResNet50 | 92–95% |
| MobileNetV3-Large | 88–93% |
| ViT-Base/16 (IN-21k) | 93–96% |
| **Soft Voting Ensemble** | **95–98%** |

The ensemble consistently outperforms all individual base models due to
variance reduction through diverse error patterns (CNN vs. Transformer).

---

## Troubleshooting

**"CUDA out of memory" during training:**
- Reduce `BATCH_SIZE` from 32 to 16 in `config.py`.
- Ensure no other GPU processes are running (`!nvidia-smi`).

**"Layer not found" in GradCAM:**
- Run `print(dict(model.named_modules()).keys())` to inspect layer names.
- Update `RESNET_TARGET_LAYER` or `MOBILENET_TARGET_LAYER` in `config.py`.

**Colab session keeps disconnecting:**
- Use a Colab Pro subscription for longer runtimes.
- All checkpoints are saved to Drive — re-run `train.py` to continue.

**ViT downloads slowly:**
- The model is downloaded from HuggingFace Hub the first time (~330 MB).
- Subsequent runs load from the HuggingFace cache automatically.
