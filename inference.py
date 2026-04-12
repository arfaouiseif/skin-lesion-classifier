import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

NUM_CLASSES = 8
IMAGE_SIZE  = 224
CHECKPOINT  = "best_model_isic2019.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    "mel":  "Melanoma",                  # index 0
    "nv":   "Melanocytic nevi",          # index 1
    "bcc":  "Basal cell carcinoma",      # index 2
    "ak":   "Actinic keratosis",         # index 3
    "bkl":  "Benign keratosis",          # index 4
    "df":   "Dermatofibroma",            # index 5
    "vasc": "Vascular lesion",           # index 6
    "scc":  "Squamous cell carcinoma",   # index 7
}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASS_NAMES.keys())}

RISK_LEVEL = {
    "nv":   ("Benign",        "#1D9E75"),
    "mel":  ("High risk",     "#E24B4A"),
    "bkl":  ("Benign",        "#1D9E75"),
    "bcc":  ("High risk",     "#E24B4A"),
    "ak":   ("Moderate risk", "#EF9F27"),
    "vasc": ("Benign",        "#1D9E75"),
    "df":   ("Benign",        "#1D9E75"),
    "scc":  ("High risk",     "#E24B4A"),
}

CLASS_INFO = {
    "nv":   "A common benign mole. Most melanocytic nevi are harmless but should be monitored for changes.",
    "mel":  "Melanoma is the most serious form of skin cancer. Immediate dermatologist consultation is strongly recommended.",
    "bkl":  "A non-cancerous skin growth that becomes more common with age. Generally harmless.",
    "bcc":  "Basal cell carcinoma is the most common skin cancer. Rarely spreads but requires treatment.",
    "ak":   "A precancerous lesion caused by sun damage. Can develop into squamous cell carcinoma if untreated.",
    "vasc": "A benign vascular skin lesion such as an angioma or hemangioma. Usually harmless.",
    "df":   "A common benign skin growth. Harmless and usually requires no treatment.",
    "scc":  "Squamous cell carcinoma is a common skin cancer that can spread if untreated. Immediate dermatologist consultation is recommended.",
}

# ── TRANSFORMS ────────────────────────────────────────────────────────────────

# standard preprocessing — used for Grad-CAM (needs clean input)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# TTA augmentations — 5 different views of the same image
# each one is a slightly different version the model votes on
TTA_TRANSFORMS = [
    # 1. clean original — always included as baseline
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 2. horizontal flip
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 3. vertical flip
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 4. slight rotation
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=(15, 15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 5. center crop then resize — focuses on lesion center
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# ── MODEL LOADER ──────────────────────────────────────────────────────────────

def load_model(checkpoint=CHECKPOINT):
    model    = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feats, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded from {checkpoint} on {DEVICE}")
    return model

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_image(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input)
    return img.convert("RGB")


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    img  = tensor.clone().squeeze(0)
    img  = img * std[:, None, None] + mean[:, None, None]
    img  = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)

# ── TTA PREDICT ───────────────────────────────────────────────────────────────

def predict_tta(model, raw_img):
    """
    Runs the model 5 times on different versions of the same image
    then averages the probabilities.

    Why this works: a skin lesion looks the same whether flipped,
    rotated, or cropped slightly differently. Averaging 5 predictions
    reduces variance and gives a more stable, accurate result.
    It's essentially a free ensemble — no retraining needed.
    """
    all_probs = []

    with torch.no_grad():
        for tta_transform in TTA_TRANSFORMS:
            tensor  = tta_transform(raw_img).unsqueeze(0).to(DEVICE)
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

    # average probabilities across all 5 augmentations
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs

# ── GRAD-CAM ──────────────────────────────────────────────────────────────────

def get_gradcam(model, image_tensor, target_class):
    target_layers = [model.layer4[-1]]
    cam           = GradCAM(model=model, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor.to(DEVICE), targets=targets)
    return grayscale_cam[0]

# ── MAIN PREDICT FUNCTION ─────────────────────────────────────────────────────

def predict(model, image_input):
    """
    Full inference pipeline with TTA.

    1. Load image
    2. Run TTA — 5 augmented predictions → average probabilities
    3. Pick predicted class from averaged probs
    4. Run Grad-CAM on the clean original image
    5. Return everything the app needs
    """
    # 1. load
    raw_img = load_image(image_input)

    # 2. TTA
    avg_probs  = predict_tta(model, raw_img)
    pred_idx   = int(np.argmax(avg_probs))
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = float(avg_probs[pred_idx])

    # 3. Grad-CAM on clean image only (unaugmented = cleaner heatmap)
    clean_tensor  = preprocess(raw_img).unsqueeze(0)
    grayscale_cam = get_gradcam(model, clean_tensor, pred_idx)

    # 4. overlay
    rgb_img     = denormalize(clean_tensor.cpu())
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 5. result dict
    all_probs_dict = {cls: round(float(avg_probs[i]), 4)
                      for i, cls in enumerate(CLASS_NAMES.keys())}

    risk, color = RISK_LEVEL[pred_class]

    return {
        "predicted_class" : pred_class,
        "full_name"       : CLASS_NAMES[pred_class],
        "confidence"      : confidence,
        "risk_level"      : risk,
        "risk_color"      : color,
        "description"     : CLASS_INFO[pred_class],
        "all_probs"       : all_probs_dict,
        "gradcam_overlay" : cam_overlay,
        "original_image"  : (rgb_img * 255).astype(np.uint8),
        "tta_used"        : True,
        "tta_count"       : len(TTA_TRANSFORMS),
    }

# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import build_path_lookup, IMG_DIRS, META_PATH, RANDOM_SEED
    from sklearn.model_selection import train_test_split
    import pandas as pd

    model = load_model()

    meta_path   = META_PATH if os.path.exists(META_PATH) else META_PATH + ".csv"
    df          = pd.read_csv(meta_path)
    path_lookup = build_path_lookup(IMG_DIRS)
    _, val_df   = train_test_split(df, test_size=0.2,
                                   stratify=df["dx"], random_state=RANDOM_SEED)

    print("─── TTA predictions ──────────────────────────────────")
    for cls in ["nv", "mel", "bcc", "bkl"]:
        subset   = val_df[val_df["dx"] == cls]
        row      = subset.sample(1, random_state=42).iloc[0]
        img_path = path_lookup[row["image_id"]]
        result   = predict(model, img_path)

        correct = "✓" if result["predicted_class"] == cls else "✗"
        print(f"\nTrue: {cls:6s} | "
              f"Pred: {result['predicted_class']:6s} | "
              f"Conf: {result['confidence']*100:.1f}% | {correct}")
        print("  Probs:", {k: f"{v*100:.1f}%"
                           for k, v in result["all_probs"].items()})

    print("\nTTA inference pipeline ready.")