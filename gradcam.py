"""
gradcam.py — Explainable AI via Grad-CAM (CNNs) and Attention Maps (ViT).

Grad-CAM (Selvaraju et al., 2017)
----------------------------------
1. Register forward and backward hooks on the target convolutional layer.
2. On a forward pass, cache the activation feature maps.
3. On a backward pass (w.r.t. the target class score), cache the gradients.
4. Global-average-pool the gradients → neuron importance weights.
5. Weight the activation maps by those importance weights and sum across channels.
6. Apply ReLU → only keep regions that positively influence the prediction.
7. Upsample the resulting coarse heatmap to the original image dimensions.
8. Overlay on the original image using a jet colourmap.

ViT Attention Maps
------------------
The HuggingFace ViTForImageClassification returns raw attention tensors
(one per block, shape: batch × heads × tokens × tokens) when
output_attentions=True.  We extract the final block's CLS-to-patch attention,
average across attention heads, and reshape into a 2D spatial grid.

Usage
-----
    from gradcam import visualise_explanations
    visualise_explanations(
        image_path, resnet, mobilenet, vit, class_names
    )
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation used by all three models
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> tuple:
    """
    Load an image from disk and return:
      - tensor : torch.Tensor shape (1, 3, 224, 224) on DEVICE
      - orig   : PIL.Image (RGB, original size) for overlay rendering
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    orig   = Image.open(image_path).convert("RGB")
    tensor = transform(orig).unsqueeze(0).to(DEVICE)
    return tensor, orig


def preprocess_pil(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL Image directly (used by the Streamlit app)."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return transform(pil_image).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Post-hoc Grad-CAM for any CNN with named modules.

    Parameters
    ----------
    model             : nn.Module — must be in eval() mode before calling generate()
    target_layer_name : str       — e.g. "layer4" for ResNet50, "features.16" for MobileNetV3
    """

    def __init__(self, model, target_layer_name: str):
        self.model       = model.eval()
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str) -> None:
        # Retrieve the target module via its dotted path name
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(
                f"[GradCAM] Layer '{layer_name}' not found in model. "
                f"Available layers:\n"
                + "\n".join(dict(self.model.named_modules()).keys())
            )

        def _forward_hook(module, inp, output):
            self.activations = output.detach()

        def _backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(_forward_hook))
        self._hooks.append(target.register_full_backward_hook(_backward_hook))

    def remove_hooks(self) -> None:
        """Call this to avoid memory leaks after generating the heatmap."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : shape (1, 3, H, W)
        class_idx    : target class index; defaults to argmax of the prediction

        Returns
        -------
        cam       : np.ndarray  shape (h, w)  values in [0, 1]
        class_idx : int
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backpropagate only the score for the target class
        score = output[0, class_idx]
        score.backward()

        # Global average pool the gradients → channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# ViT Attention Map
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_vit_attention_map(vit_model, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract a spatial attention map from the last ViT encoder block.

    The CLS token attends to all patch tokens; averaging this attention
    across heads yields a rough spatial salience map.

    Returns
    -------
    cam : np.ndarray  shape (grid, grid)  values in [0, 1]
    """
    vit_model.eval()
    # Ensure config allows attentions (required for some transformers versions)
    vit_model.config.output_attentions = True
    
    outputs = vit_model(
        pixel_values=input_tensor,
        output_attentions=True,
    )

    # Safety check: if attentions are missing (e.g. using SDPA), return a blank map
    if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
        print("[Warning] ViT model did not return attentions. Returning blank map.")
        grid_size = config.IMAGE_SIZE // 16  # Default patch size for ViT-B/16
        return np.zeros((grid_size, grid_size))

    # attentions[-1] : (batch, heads, n_tokens, n_tokens)
    last_attn = outputs.attentions[-1]       # final encoder block
    cls_attn  = last_attn[0, :, 0, 1:]      # CLS → patch tokens: (heads, patches)
    cls_attn  = cls_attn.mean(dim=0)         # average over heads → (patches,)

    n_patches = cls_attn.shape[0]
    grid_size = int(n_patches ** 0.5)        # 14 for ViT-Base/16 with 224×224

    cam = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 1e-8:
        cam = cam / cam.max()

    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helper
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(
    cam: np.ndarray,
    original_img: Image.Image,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Bilinearly upsample `cam` to IMAGE_SIZE × IMAGE_SIZE, apply jet colourmap,
    and blend with the original image.

    Returns
    -------
    overlay : np.ndarray  uint8  shape (IMAGE_SIZE, IMAGE_SIZE, 3)
    """
    size   = config.IMAGE_SIZE
    orig   = np.array(original_img.resize((size, size)))           # H×W×3 uint8

    # Upsample the low-resolution CAM
    heatmap = cv2.resize(cam, (size, size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)         # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)             # → RGB

    overlay = (alpha * heatmap + (1.0 - alpha) * orig).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Full multi-model visualisation
# ─────────────────────────────────────────────────────────────────────────────
def visualise_explanations(
    image_path: str,
    resnet,
    mobilenet,
    vit,
    class_names: list,
    true_label: str = None,
    save_path: str = "gradcam_explanation.png",
) -> None:
    """
    Generate Grad-CAM / attention heatmaps for all three models and display
    them side-by-side alongside the ensemble prediction.

    Parameters
    ----------
    image_path  : path to the leaf image file
    resnet      : trained ResNet50 (eval mode)
    mobilenet   : trained MobileNetV3 (eval mode)
    vit         : trained ViT (eval mode)
    class_names : list of class name strings
    true_label  : optional ground truth string for the figure title
    save_path   : where to save the figure
    """
    input_tensor, orig_img = preprocess_image(image_path)

    # ── ResNet50 Grad-CAM ────────────────────────────────────────────────────
    gcam_r            = GradCAM(resnet, config.RESNET_TARGET_LAYER)
    cam_r, pred_r_idx = gcam_r.generate(input_tensor)
    gcam_r.remove_hooks()
    overlay_r         = overlay_heatmap(cam_r, orig_img)

    # ── MobileNetV3 Grad-CAM ─────────────────────────────────────────────────
    gcam_m            = GradCAM(mobilenet, config.MOBILENET_TARGET_LAYER)
    cam_m, pred_m_idx = gcam_m.generate(input_tensor)
    gcam_m.remove_hooks()
    overlay_m         = overlay_heatmap(cam_m, orig_img)

    # ── ViT Attention Map ────────────────────────────────────────────────────
    cam_v     = get_vit_attention_map(vit, input_tensor)
    overlay_v = overlay_heatmap(cam_v, orig_img)

    with torch.no_grad():
        pred_v_idx = vit(pixel_values=input_tensor).logits.argmax(dim=1).item()

    # ── Ensemble soft vote ───────────────────────────────────────────────────
    with torch.no_grad():
        prob_r   = F.softmax(resnet(input_tensor), dim=1)
        prob_m   = F.softmax(mobilenet(input_tensor), dim=1)
        prob_v   = F.softmax(vit(pixel_values=input_tensor).logits, dim=1)
        avg_prob = (prob_r + prob_m + prob_v) / 3.0

    pred_ens = avg_prob.argmax(dim=1).item()
    conf_ens = avg_prob.max().item() * 100

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    size = config.IMAGE_SIZE
    orig_resized = np.array(orig_img.resize((size, size)))

    panels = [
        (overlay_r,    f"ResNet50\n→ {class_names[pred_r_idx]}"),
        (overlay_m,    f"MobileNetV3\n→ {class_names[pred_m_idx]}"),
        (overlay_v,    f"ViT Attention\n→ {class_names[pred_v_idx]}"),
        (orig_resized, f"Ensemble (Soft Vote)\n→ {class_names[pred_ens]}  "
                       f"({conf_ens:.1f}%)"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")

    if true_label is not None:
        fig.suptitle(
            f"Ground Truth: {true_label}",
            fontsize=13, color="#27ae60", fontweight="bold", y=1.02,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[gradcam] Explanation saved → {save_path}")
