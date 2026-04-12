import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DATA_DIR,
    IMG_DIRS,
    META_PATH,
    val_transforms,
)

import pandas as pd
from sklearn.model_selection import train_test_split
from data import build_path_lookup, RANDOM_SEED

# ── CONFIG ────────────────────────────────────────────────────────────────────

NUM_CLASSES    = 7
CHECKPOINT     = "best_model_finetuned.pth"   # our best model from day 3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES    = 8                             # how many images to visualize

print(f"Running on: {DEVICE}")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    model    = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feats, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# ── PREDICT SINGLE IMAGE ──────────────────────────────────────────────────────

def predict(model, image_tensor):
    """
    Takes a single preprocessed image tensor [1, 3, 224, 224]
    Returns predicted class index and all class probabilities
    """
    with torch.no_grad():
        outputs     = model(image_tensor.to(DEVICE))         # raw logits [1, 7]
        probs       = torch.softmax(outputs, dim=1)          # convert to probabilities
        pred_idx    = probs.argmax(dim=1).item()             # index of highest prob
        pred_prob   = probs[0, pred_idx].item()              # confidence score
    return pred_idx, pred_prob, probs[0].cpu().numpy()

# ── GRAD-CAM ──────────────────────────────────────────────────────────────────

def get_gradcam(model, image_tensor, target_class):
    """
    Generates a Grad-CAM heatmap for a given image and target class.
    target_class: the class index we want to explain (usually the predicted class)
    """
    # layer4 is the last convolutional layer in ResNet50
    # it has the richest spatial information — perfect for Grad-CAM
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # ClassifierOutputTarget tells Grad-CAM which class to explain
    targets = [ClassifierOutputTarget(target_class)]

    # generate the heatmap — output is a 2D array of shape [224, 224]
    # values between 0 and 1, where 1 = most important region
    grayscale_cam = cam(input_tensor=image_tensor.to(DEVICE), targets=targets)
    grayscale_cam = grayscale_cam[0]   # remove batch dimension → [224, 224]

    return grayscale_cam

# ── DENORMALIZE IMAGE ─────────────────────────────────────────────────────────

def denormalize(tensor):
    """
    Reverses the ImageNet normalization so we can display the original image.
    Without this the image would look completely wrong (negative values etc.)
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    img = tensor.clone().squeeze(0)        # remove batch dim → [3, 224, 224]
    img = img * std[:, None, None] + mean[:, None, None]   # reverse normalize
    img = img.permute(1, 2, 0).numpy()    # [3,224,224] → [224,224,3]
    img = np.clip(img, 0, 1)              # clamp to [0, 1]
    return img

# ── VISUALIZE GRID ────────────────────────────────────────────────────────────

def visualize_gradcam(model, samples, path_lookup):
    """
    For each sample image:
      - show original image
      - show Grad-CAM overlay
      - show confidence bar chart
    Saves everything as a single grid PNG.
    """
    n       = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 4))
    fig.suptitle("Grad-CAM — what the model sees", fontsize=14, y=1.01)

    for i, (image_id, true_label) in enumerate(samples):
        # ── load and preprocess image ──────────────────────────────────────
        img_path     = path_lookup[image_id]
        raw_img      = Image.open(img_path).convert("RGB")
        image_tensor = val_transforms(raw_img).unsqueeze(0)  # add batch dim → [1,3,224,224]

        # ── predict ───────────────────────────────────────────────────────
        pred_idx, pred_prob, all_probs = predict(model, image_tensor)
        pred_class = IDX_TO_CLASS[pred_idx]
        true_class = true_label

        # ── grad-cam heatmap ──────────────────────────────────────────────
        grayscale_cam = get_gradcam(model, image_tensor, pred_idx)

        # overlay heatmap on the original image
        rgb_img     = denormalize(image_tensor)
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # ── column 1: original image ──────────────────────────────────────
        ax1 = axes[i, 0]
        ax1.imshow(raw_img)
        correct = "✓" if pred_class == true_class else "✗"
        color   = "green" if pred_class == true_class else "red"
        ax1.set_title(f"True: {CLASS_NAMES[true_class]}", fontsize=9)
        ax1.axis("off")

        # ── column 2: grad-cam overlay ────────────────────────────────────
        ax2 = axes[i, 1]
        ax2.imshow(cam_overlay)
        ax2.set_title(
            f"Pred: {CLASS_NAMES[pred_class]} ({pred_prob*100:.1f}%) {correct}",
            fontsize=9, color=color
        )
        ax2.axis("off")

        # ── column 3: confidence bar chart ────────────────────────────────
        ax3    = axes[i, 2]
        labels = list(CLASS_NAMES.keys())
        colors = ["#E24B4A" if l == pred_class else "#B5D4F4" for l in labels]
        bars   = ax3.barh(labels, all_probs * 100, color=colors)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Confidence (%)", fontsize=8)
        ax3.tick_params(labelsize=8)
        ax3.set_title("Class probabilities", fontsize=9)

        # add percentage labels on bars
        for bar, prob in zip(bars, all_probs):
            if prob > 0.03:
                ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         f"{prob*100:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig("gradcam_results.png", dpi=150, bbox_inches="tight")
    print("Saved gradcam_results.png")
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load model
    model = load_model()
    print(f"Model loaded from {CHECKPOINT}\n")

    # load metadata and pick sample images from the val set
    meta_path   = META_PATH if os.path.exists(META_PATH) else META_PATH + ".csv"
    df          = pd.read_csv(meta_path)
    path_lookup = build_path_lookup(IMG_DIRS)

    _, val_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=RANDOM_SEED
    )

    # pick one sample from each class so we see all 7 + one extra
    samples = []
    for cls in CLASS_NAMES.keys():
        subset = val_df[val_df["dx"] == cls]
        if len(subset) > 0:
            row = subset.sample(1, random_state=42).iloc[0]
            samples.append((row["image_id"], row["dx"]))

    print(f"Running Grad-CAM on {len(samples)} images (one per class)...\n")

    # run and visualize
    visualize_gradcam(model, samples, path_lookup)

    # also test the predict function standalone
    print("\n─── Standalone prediction test ───────────────────")
    row          = val_df.sample(1, random_state=99).iloc[0]
    img_path     = path_lookup[row["image_id"]]
    raw_img      = Image.open(img_path).convert("RGB")
    image_tensor = val_transforms(raw_img).unsqueeze(0)

    pred_idx, pred_prob, all_probs = predict(model, image_tensor)
    print(f"Image     : {row['image_id']}")
    print(f"True label: {row['dx']} ({CLASS_NAMES[row['dx']]})")
    print(f"Predicted : {IDX_TO_CLASS[pred_idx]} ({CLASS_NAMES[IDX_TO_CLASS[pred_idx]]}) — {pred_prob*100:.1f}% confident")
    print("\nAll class probabilities:")
    for cls, prob in zip(CLASS_NAMES.keys(), all_probs):
        bar = "█" * int(prob * 40)
        print(f"  {cls:6s}: {bar:<40s} {prob*100:.1f}%")

    print("\nDay 4 complete!")