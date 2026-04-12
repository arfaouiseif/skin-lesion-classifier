import streamlit as st
import numpy as np
from PIL import Image
import torch
import sys
import os

# add current directory to path so we can import our modules
sys.path.append(os.path.dirname(__file__))
from inference import load_model, predict, CLASS_NAMES, RISK_LEVEL, CHECKPOINT

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: 600;
        color: white;
        margin-bottom: 10px;
    }
    .confidence-text {
        font-size: 42px;
        font-weight: 700;
        margin: 0;
    }
    .class-name {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .description-box {
        background-color: #f8f9fa;
        border-left: 4px solid #378ADD;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 14px;
        color: #444;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border: 1px solid #EF9F27;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #633806;
        margin-top: 16px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL (cached — only runs once) ──────────────────────────────────────

@st.cache_resource
def get_model():
    return load_model(CHECKPOINT)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 About")
    st.markdown("""
    This app uses a **ResNet50** deep learning model fine-tuned on the
    **ISIC2019 dataset** (25,533 dermoscopy images) to classify skin lesions
    into 8 categories.

    **Model performance:**
    - Overall accuracy: **81.4%**
    - Macro F1 score: **0.76**
    - Weighted F1: **0.81**

    **What is Grad-CAM?**
    The heatmap overlay shows *where* the model is looking to make its
    decision. Red regions = high importance, blue = low importance.
    """)

    st.markdown("---")
    st.markdown("**Detectable conditions:**")
    for cls, name in CLASS_NAMES.items():
        risk, color = RISK_LEVEL[cls]
        st.markdown(
            f'<span style="color:{color};">●</span> **{cls}** — {name}',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-box'>"
        "⚠️ <b>Medical disclaimer</b><br>"
        "This tool is for educational purposes only and does <b>not</b> "
        "constitute medical advice. Always consult a qualified dermatologist "
        "for diagnosis and treatment."
        "</div>",
        unsafe_allow_html=True
    )

# ── MAIN PAGE ─────────────────────────────────────────────────────────────────

st.title("🔬 Skin Lesion Risk Classifier")
st.markdown("Upload a dermoscopy image to get an AI-powered classification with visual explanation.")

# ── UPLOAD ────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose a dermoscopy image",
    type=["jpg", "jpeg", "png"],
    help="Upload a close-up or dermoscopy image of a skin lesion"
)

# ── SAMPLE IMAGES ─────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.markdown("---")
    st.markdown("#### Don't have an image? Try a sample:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**How to use:**")
        st.markdown("""
        1. Upload a dermoscopy image using the uploader above
        2. Wait a few seconds for the model to analyze it
        3. View the prediction, confidence score, and Grad-CAM heatmap
        4. Read the condition description and risk level
        """)
    with col2:
        st.markdown("**Risk levels:**")
        st.markdown("""
        - 🟢 **Benign** — generally harmless, monitor for changes
        - 🟡 **Moderate risk** — precancerous, consult a doctor
        - 🔴 **High risk** — requires immediate medical attention
        """)
    with col3:
        st.markdown("**Model info:**")
        st.markdown("""
        - Architecture: ResNet50
        - Dataset: HAM10000
        - Training: Transfer learning + fine-tuning
        - Classes: 7 skin lesion types
        """)

# ── PREDICTION ────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    model = get_model()

    with st.spinner("Analyzing image..."):
        result = predict(model, uploaded_file)

    st.markdown("---")

    # ── TOP ROW: images ───────────────────────────────────────────────────────
    st.markdown("### Results")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("<div class='section-header'>Original image</div>",
                    unsafe_allow_html=True)
        st.image(result["original_image"], use_container_width=True)

    with img_col2:
        st.markdown("<div class='section-header'>Grad-CAM — what the model sees</div>",
                    unsafe_allow_html=True)
        st.image(result["gradcam_overlay"],use_container_width=True)
        st.caption("🔴 Red = regions that most influenced the prediction")

    st.markdown("---")

    # ── MIDDLE ROW: prediction details ────────────────────────────────────────
    detail_col1, detail_col2 = st.columns([1, 1])

    with detail_col1:
        st.markdown("#### Prediction")

        # risk badge
        risk  = result["risk_level"]
        color = result["risk_color"]
        st.markdown(
            f'<div class="risk-badge" style="background-color:{color};">'
            f'{risk}</div>',
            unsafe_allow_html=True
        )

        # class name and confidence
        st.markdown(
            f'<p class="class-name">{result["full_name"]}'
            f' <span style="color:#888;font-size:16px;">({result["predicted_class"]})</span></p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="confidence-text" style="color:{color};">'
            f'{result["confidence"]*100:.1f}%</p>',
            unsafe_allow_html=True
        )
        st.caption("model confidence")
        st.caption("✓ 5-pass test time augmentation active")


        # description
        st.markdown(
            f'<div class="description-box">{result["description"]}</div>',
            unsafe_allow_html=True
        )

    with detail_col2:
        st.markdown("#### All class probabilities")

        # bar chart for all 7 classes
        for cls, prob in sorted(result["all_probs"].items(),
                                key=lambda x: x[1], reverse=True):
            _, bar_color = RISK_LEVEL[cls]
            is_pred      = cls == result["predicted_class"]
            label        = f"**{CLASS_NAMES[cls]}**" if is_pred else CLASS_NAMES[cls]

            col_label, col_bar = st.columns([2, 3])
            with col_label:
                st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            with col_bar:
                st.progress(float(prob), text=f"{prob*100:.1f}%")

    st.markdown("---")

    # ── BOTTOM: disclaimer ────────────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer-box'>"
        "⚠️ <b>Important medical disclaimer:</b> This AI tool is intended for "
        "educational and research purposes only. It does <b>not</b> replace "
        "professional medical advice, diagnosis, or treatment. If you have "
        "concerns about a skin lesion, please consult a qualified dermatologist "
        "immediately. Never delay seeking medical advice because of something "
        "you read or saw in this application."
        "</div>",
        unsafe_allow_html=True
    )