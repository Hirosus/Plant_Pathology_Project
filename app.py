"""
app.py — Streamlit deployment interface for the Plant Pathology Diagnosis System.

Run locally:
    streamlit run app.py

Run on Colab (via ngrok — see README.md):
    !pip install pyngrok -q
    from pyngrok import ngrok
    !streamlit run app.py &
    print(ngrok.connect(8501))
"""

import io
import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")  # Non-interactive backend for server environments

import config
from gradcam import (
    GradCAM,
    get_vit_attention_map,
    overlay_heatmap,
    preprocess_pil,
)
from models.ensemble import SoftVotingEnsemble
from models.mobilenet_model import build_mobilenetv3
from models.resnet_model import build_resnet50
from models.vit_model import build_vit
from utils import load_class_names

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached — loads only once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — please wait…")
def load_all_models():
    class_names = load_class_names()
    num_classes = len(class_names)

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

    ensemble = SoftVotingEnsemble(resnet, mobilenet, vit).to(DEVICE)
    ensemble.eval()

    return resnet, mobilenet, vit, ensemble, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(cls_name: str) -> str:
    """Format class name for display: Tomato___Early_blight → Tomato — Early Blight"""
    return cls_name.replace("___", " — ").replace("_", " ").title()


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0A0F0D")
    buf.seek(0)
    return Image.open(buf).copy()


def _hero_header():
    st.markdown("""
    <div class="animate-fade-up" style="padding:2.5rem 0 1.5rem 0; border-bottom:1px solid #1E3028; margin-bottom:2rem;">
        <div class="interactive-pill" style="display:inline-block; background:rgba(34,197,94,0.1);
            border:1px solid rgba(34,197,94,0.3); border-radius:20px;
            padding:0.25rem 0.85rem; font-size:0.8rem;
            font-family:'JetBrains Mono',monospace; color:#22C55E;
            letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1rem;">
            ◉ System Active — v1.0
        </div>
        <h1 style="font-family:'DM Serif Display',Georgia,serif; font-size:3.5rem;
            font-weight:400; color:#F0FDF4; margin:0 0 0.5rem 0;
            line-height:1.1; letter-spacing:-0.02em;">
            Plant<span style="color:#22C55E;">Dx</span>
            <span style="font-style:italic; color:#86EFAC; font-size:2.2rem;">
                Diagnostic Engine</span>
        </h1>
        <p style="font-family:'DM Sans',sans-serif; color:#4B7A5E;
            font-size:1.05rem; margin:0; letter-spacing:0.02em;">
            Heterogeneous Ensemble &nbsp;·&nbsp; ResNet50 + MobileNetV3 + ViT-Base/16
            &nbsp;·&nbsp; Soft Voting Fusion &nbsp;·&nbsp; Grad-CAM XAI
        </p>
    </div>
    """, unsafe_allow_html=True)


def _section_label(text, icon=""):
    st.markdown(f"""
    <div class="animate-fade-up" style="display:flex; align-items:center; gap:0.6rem;
        margin-bottom:1rem; margin-top:0.5rem; opacity:0; animation-delay:0.1s;">
        <span style="font-size:1.25rem;">{icon}</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;
            letter-spacing:0.14em; text-transform:uppercase; color:#4B7A5E;">
            {text}</span>
        <div style="flex:1; height:1px; background:#1E3028; margin-left:0.5rem;"></div>
    </div>
    """, unsafe_allow_html=True)


def _result_card(disease_name, confidence, is_healthy, model_preds):
    colour    = "#22C55E" if is_healthy else "#EF4444"
    bg_colour = "rgba(34,197,94,0.08)" if is_healthy else "rgba(239,68,68,0.08)"
    border_c  = "rgba(34,197,94,0.3)"  if is_healthy else "rgba(239,68,68,0.3)"
    icon      = "✦" if is_healthy else "⚠"
    status    = "HEALTHY" if is_healthy else "DISEASE DETECTED"
    bar_width = int(confidence)

    preds_html = "".join([
        f'<div class="interactive-pill" style="display:flex; justify-content:space-between; padding:0.5rem 0.25rem;'
        f'border-bottom:1px solid #1A2820; font-size:0.95rem;">'
        f'<span style="color:#4B7A5E; font-family:\'JetBrains Mono\',monospace;">{m}</span>'
        f'<span style="color:#86EFAC; font-family:\'JetBrains Mono\',monospace;'
        f'font-weight:500;">{v:.1f}%</span></div>'
        for m, v in model_preds
    ])

    st.markdown(f"""
    <div class="animate-fade-up hover-card" style="background:{bg_colour}; border:1.5px solid {border_c};
        border-radius:16px; padding:1.75rem; margin-bottom:1rem; opacity:0; animation-delay:0.2s;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
            letter-spacing:0.18em; color:{colour}; text-transform:uppercase;
            margin-bottom:0.5rem;">{icon} {status}</div>
        <div style="font-family:'DM Serif Display',Georgia,serif; font-size:2.25rem;
            color:#F0FDF4; margin-bottom:0.25rem; line-height:1.15;">
            {disease_name}</div>
        <div style="margin:1.25rem 0 0.5rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem;
                margin-bottom:0.4rem; font-family:'JetBrains Mono',monospace;">
                <span style="color:#4B7A5E;">ENSEMBLE CONFIDENCE</span>
                <span style="color:{colour}; font-weight:600;">{confidence:.1f}%</span>
            </div>
            <div style="background:#0A0F0D; border-radius:999px; height:8px;
                overflow:hidden; border:1px solid #1E3028;">
                <div style="width:{bar_width}%; height:100%;
                    background:linear-gradient(90deg,{colour}88,{colour});
                    border-radius:999px;"></div>
            </div>
        </div>
        <div style="margin-top:1.25rem;">
            <div style="font-size:0.75rem; letter-spacing:0.12em; text-transform:uppercase;
                color:#2A4A38; font-family:'JetBrains Mono',monospace;
                margin-bottom:0.5rem;">Individual Model Votes</div>
            {preds_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _stat_pill(label, value):
    st.markdown(f"""
    <div class="interactive-pill" style="display:flex; justify-content:space-between; align-items:center;
        padding:0.5rem 0.25rem; border-bottom:1px solid #1A2820;">
        <span style="font-size:0.85rem; color:#4B7A5E;">{label}</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
            color:#86EFAC; background:rgba(34,197,94,0.08);
            padding:0.15rem 0.5rem; border-radius:4px;">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def _heatmap_card(img_array, title, subtitle):
    st.markdown(f"""
    <div class="animate-fade-up hover-card" style="background:#111916; border:1px solid #1E3028; border-radius:14px;
        overflow:hidden; margin-bottom:0.5rem; opacity:0; animation-delay:0.3s;">
        <div style="padding:0.6rem 0.9rem; border-bottom:1px solid #1A2820;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;
                color:#86EFAC; font-weight:500;">{title}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                color:#2A4A38; margin-top:0.1rem;">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.image(img_array, use_container_width=True)


def _feature_card(icon, title, desc):
    st.markdown(f"""
    <div class="animate-fade-up hover-card" style="background:#111916; border:1px solid #1E3028; border-radius:16px;
        padding:1.5rem; height:100%; opacity:0; animation-delay:0.1s;">
        <div class="interactive-pill" style="font-size:2.2rem; margin-bottom:0.75rem; display:inline-block;">{icon}</div>
        <div style="font-family:'DM Serif Display',Georgia,serif; font-size:1.3rem;
            color:#F0FDF4; margin-bottom:0.5rem;">{title}</div>
        <div style="font-size:0.95rem; color:#4B7A5E; line-height:1.65;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def _step_card(n, title, desc):
    st.markdown(f"""
    <div class="animate-fade-up hover-card" style="background:#111916; border:1px solid #1E3028;
        border-radius:12px; padding:1.1rem; opacity:0; animation-delay:0.2s;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
            color:#22C55E; letter-spacing:0.15em; margin-bottom:0.4rem;">
            STEP {n}</div>
        <div style="font-size:1rem; color:#86EFAC; margin-bottom:0.3rem;
            font-weight:500;">{title}</div>
        <div style="font-size:0.9rem; color:#4B7A5E; line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Pathology Diagnosis System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0A0F0D;
    --bg-card:       #111916;
    --border:        #1E3028;
    --border-bright: #2A4A38;
    --green-primary: #22C55E;
    --text-primary:  #F0FDF4;
    --text-secondary:#86EFAC;
    --text-muted:    #4B7A5E;
}

@keyframes fadeUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.animate-fade-up {
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.hover-card {
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.hover-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.12) !important;
    border-color: rgba(34, 197, 94, 0.4) !important;
}

.interactive-pill {
    transition: all 0.2s ease;
}
.interactive-pill:hover {
    background: rgba(34, 197, 94, 0.15) !important;
    transform: scale(1.05);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0A0F0D !important;
    color: #F0FDF4 !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background-color: transparent !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1610 0%, #0A0F0D 100%) !important;
    border-right: 1px solid #1E3028 !important;
}

[data-testid="stFileUploader"] {
    background: #111916 !important;
    border: 1.5px dashed #2A4A38 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #16A34A 0%, #15803D 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(34,197,94,0.4) !important;
}

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1.5px solid #2A4A38 !important;
    color: #86EFAC !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #22C55E !important;
    color: #22C55E !important;
    background: rgba(34,197,94,0.06) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #22C55E !important;
}

[data-testid="caption"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #4B7A5E !important;
    text-align: center !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

hr { border-color: #1E3028 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0F0D; }
::-webkit-scrollbar-thumb { background: #2A4A38; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0.5rem 1.5rem 0.5rem;">
        <div style="font-family:'DM Serif Display',Georgia,serif; font-size:1.6rem;
            color:#F0FDF4; margin-bottom:0.2rem;">
            Plant<span style="color:#22C55E;">Dx</span></div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
            color:#2A4A38; letter-spacing:0.14em; text-transform:uppercase;">
            Pathology Diagnostic Engine</div>
        <div style="height:1px; background:#1E3028; margin:1rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.7rem; letter-spacing:0.1em;
        text-transform:uppercase; color:#2A4A38;
        font-family:'JetBrains Mono',monospace; margin-bottom:0.75rem;">
        Upload Specimen</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("⟶  Run Diagnosis", use_container_width=True)

    st.markdown("""
    <div style="height:1px; background:#1E3028; margin:1rem 0;"></div>
    <div style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
        color:#2A4A38; font-family:'JetBrains Mono',monospace;
        margin-bottom:0.75rem;">System Architecture</div>
    """, unsafe_allow_html=True)

    _stat_pill("ResNet50",    "ImageNet-1k V2")
    _stat_pill("MobileNetV3", "Lightweight CNN")
    _stat_pill("ViT-Base/16", "ImageNet-21k")
    _stat_pill("Fusion",      "Soft Voting")
    _stat_pill("XAI",         "Grad-CAM + Attn")

    st.markdown("""
    <div style="height:1px; background:#1E3028; margin:1rem 0;"></div>
    <div style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
        color:#2A4A38; font-family:'JetBrains Mono',monospace;
        margin-bottom:0.75rem;">Supported Crops</div>
    """, unsafe_allow_html=True)

    for crop in ["🍅  Tomato", "🥔  Potato", "🫑  Pepper"]:
        st.markdown(
            f'<div style="padding:0.35rem 0; font-size:0.82rem; color:#4B7A5E;'
            f'border-bottom:1px solid #111916;">{crop}</div>',
            unsafe_allow_html=True,
        )

# ── Main area ─────────────────────────────────────────────────────────────────
_hero_header()

if run_btn and uploaded is not None:
    # Load models
    resnet, mobilenet, vit, ensemble, class_names = load_all_models()

    # Load the uploaded image
    pil_image    = Image.open(uploaded).convert("RGB")
    input_tensor = preprocess_pil(pil_image)

    # ── Ensemble inference ────────────────────────────────────────────────────
    with st.spinner("Running ensemble inference…"):
        with torch.no_grad():
            prob_r   = F.softmax(resnet(input_tensor), dim=1)
            prob_m   = F.softmax(mobilenet(input_tensor), dim=1)
            prob_v   = F.softmax(vit(pixel_values=input_tensor).logits, dim=1)
            avg_prob = (prob_r + prob_m + prob_v) / 3.0

        ens_pred_idx  = avg_prob.argmax(dim=1).item()
        ens_confidence = avg_prob.max().item() * 100
        ens_pred_name  = class_names[ens_pred_idx]

        r_idx = prob_r.argmax(dim=1).item()
        m_idx = prob_m.argmax(dim=1).item()
        v_idx = prob_v.argmax(dim=1).item()

    # ── Results section ───────────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1.6], gap="large")
    with col_img:
        _section_label("Specimen", "🔬")
        st.image(
            pil_image.resize((320, 320)),
            caption=uploaded.name,
        )

    with col_result:
        is_healthy = "healthy" in ens_pred_name.lower()
        _section_label("Diagnosis", "⟶")
        _result_card(
            disease_name=_fmt(ens_pred_name),
            confidence=ens_confidence,
            is_healthy=is_healthy,
            model_preds=[
                ("ResNet50",    prob_r.max().item() * 100),
                ("MobileNetV3", prob_m.max().item() * 100),
                ("ViT-Base/16", prob_v.max().item() * 100),
            ],
        )

    # ── Grad-CAM heatmaps ─────────────────────────────────────────────────────
    _section_label("Explainable AI — Gradient Activation Maps", "🧬")
    st.markdown("""
    <div style="background:rgba(34,197,94,0.05); border:1px solid rgba(34,197,94,0.15);
        border-radius:10px; padding:0.75rem 1rem; font-size:0.82rem; color:#4B7A5E;
        margin-bottom:1.25rem;">
        <strong style="color:#86EFAC;">Warm / red regions</strong> indicate where each
        model focused its attention. Verify these align with visible leaf symptoms
        (lesions, necrotic rings, water-soaked spots).
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating Grad-CAM heatmaps…"):
        gcam_r          = GradCAM(resnet,    config.RESNET_TARGET_LAYER)
        cam_r, _        = gcam_r.generate(input_tensor, ens_pred_idx)
        gcam_r.remove_hooks()
        overlay_r_img   = overlay_heatmap(cam_r, pil_image)

        gcam_m          = GradCAM(mobilenet, config.MOBILENET_TARGET_LAYER)
        cam_m, _        = gcam_m.generate(input_tensor, ens_pred_idx)
        gcam_m.remove_hooks()
        overlay_m_img   = overlay_heatmap(cam_m, pil_image)

        cam_v           = get_vit_attention_map(vit, input_tensor)
        overlay_v_img   = overlay_heatmap(cam_v, pil_image)

    col_r, col_m, col_v = st.columns(3)
    with col_r:
        _heatmap_card(overlay_r_img, "ResNet50",    "Grad-CAM · Layer4")
    with col_m:
        _heatmap_card(overlay_m_img, "MobileNetV3", "Grad-CAM · Features.16")
    with col_v:
        _heatmap_card(overlay_v_img, "ViT-Base/16", "Attention Map · Block 12")

    # ── Top-5 probability chart ───────────────────────────────────────────────
    _section_label("Probability Distribution — Top 5 Classes", "📊")

    top5_probs, top5_idxs = avg_prob[0].topk(5)
    top5_names = [_fmt(class_names[i]) for i in top5_idxs.cpu().numpy()]
    top5_vals  = top5_probs.cpu().detach().numpy() * 100

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0A0F0D")
    ax.set_facecolor("#111916")
    colours = ["#22C55E"] + ["#1E3028"] * 4
    bars = ax.barh(
        top5_names[::-1], top5_vals[::-1],
        color=colours[::-1], height=0.55,
        edgecolor="#1E3028", linewidth=0.5,
    )
    ax.set_xlabel("Probability (%)", color="#4B7A5E", fontsize=9, fontfamily="monospace")
    ax.set_xlim(0, 115)
    ax.tick_params(colors="#4B7A5E", labelsize=8.5)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#1A2820", linewidth=0.5, alpha=0.8)
    for tick in ax.get_yticklabels():
        tick.set_color("#86EFAC")
        tick.set_fontfamily("monospace")
        tick.set_fontsize(8.5)
    for bar, val in zip(bars, top5_vals[::-1]):
        colour = "#22C55E" if val == top5_vals[0] else "#2A4A38"
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", fontsize=8.5, color=colour, fontfamily="monospace",
        )
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    # ── Download section ──────────────────────────────────────────────────────
    _section_label("Export Report", "⬇")

    # Composite result image
    comp_fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    comp_fig.patch.set_facecolor("#0A0F0D")
    size = config.IMAGE_SIZE
    orig_arr = np.array(pil_image.resize((size, size)))

    for ax, img, title in zip(
        axes,
        [overlay_r_img, overlay_m_img, overlay_v_img, orig_arr],
        [
            f"ResNet50\n{_fmt(class_names[r_idx])}",
            f"MobileNetV3\n{_fmt(class_names[m_idx])}",
            f"ViT Attention\n{_fmt(class_names[v_idx])}",
            f"Ensemble\n{_fmt(ens_pred_name)} ({ens_confidence:.1f}%)",
        ],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color="#86EFAC", fontfamily="monospace", pad=10)
        ax.axis("off")
        ax.set_facecolor("#0A0F0D")

    plt.tight_layout()
    comp_img = _fig_to_pil(comp_fig)
    plt.close(comp_fig)

    buf = io.BytesIO()
    comp_img.save(buf, format="PNG")
    st.download_button(
        "⬇  Download Grad-CAM Report (PNG)",
        data=buf.getvalue(),
        file_name="plant_diagnosis_report.png",
        mime="image/png",
        use_container_width=True,
    )

elif run_btn and uploaded is None:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25);
        border-radius:12px; padding:1rem 1.25rem; color:#F59E0B; font-size:0.88rem;">
        ⚠  No image uploaded. Please select a leaf image in the sidebar first.
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing screen
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        _feature_card("🧬", "Ensemble Inference",
            "Three independent deep learning models vote on every diagnosis. "
            "Soft-voting consistently outperforms any single model.")
    with col2:
        _feature_card("🔬", "Grad-CAM XAI",
            "Every diagnosis includes visual proof — heatmaps showing exactly "
            "which leaf regions drove the model's decision.")
    with col3:
        _feature_card("⚡", "Three Architectures",
            "ResNet50 (local texture) · MobileNetV3 (efficient features) · "
            "ViT-Base/16 IN-21k (global context). Diverse errors, unified output.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _section_label("How to use", "→")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _step_card(1, "Upload Image",
            "Use the sidebar to upload a clear photograph of a Tomato, Potato, or Pepper leaf.")
    with c2:
        _step_card(2, "Run Diagnosis",
            "Click Run Diagnosis to trigger the full ensemble inference pipeline.")
    with c3:
        _step_card(3, "Review Results",
            "Check the diagnosis, confidence score, and individual model predictions.")
    with c4:
        _step_card(4, "Verify via XAI",
            "Inspect the Grad-CAM heatmaps to confirm the model focused on real symptoms.")
