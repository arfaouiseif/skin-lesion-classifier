import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

from data import (
    HAM10000Dataset,
    build_path_lookup,
    val_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DATA_DIR,
    IMG_DIRS,
    META_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
CHECKPOINT  = "best_model_finetuned.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ── GET ALL PREDICTIONS ON VAL SET ───────────────────────────────────────────

def get_predictions(model, val_loader):
    """
    Runs the model on the entire val set.
    Returns:
      all_preds  — predicted class index for each image
      all_labels — true class index for each image
      all_probs  — full probability vector for each image (needed for ROC)
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)                          # [batch, 7]
            probs   = torch.softmax(outputs, dim=1)         # [batch, 7]
            preds   = probs.argmax(dim=1)                   # [batch]

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    return all_preds, all_labels, all_probs

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────

def plot_confusion_matrix(all_preds, all_labels):
    """
    Confusion matrix: rows = true class, columns = predicted class.
    Each cell shows how many images of class X were predicted as class Y.
    Perfect model = diagonal is full, everything else is 0.
    """
    cm     = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    class_labels = list(CLASS_NAMES.keys())
    fig, ax      = plt.subplots(figsize=(9, 7))

    im = ax.imshow(cm_pct, cmap="Blues")
    plt.colorbar(im, ax=ax, label="% of true class")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (% of true class)")

    # write percentage inside each cell
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")
    plt.show()

# ── PER-CLASS F1 SCORES ───────────────────────────────────────────────────────

def plot_f1_scores(all_preds, all_labels):
    """
    F1 score = harmonic mean of precision and recall.
    Precision: of everything predicted as class X, how many were actually X?
    Recall:    of all actual class X images, how many did we correctly find?
    F1 balances both — a high F1 means both precision and recall are good.
    """
    report = classification_report(
        all_labels, all_preds,
        target_names=list(CLASS_NAMES.keys()),
        output_dict=True
    )

    # print full report
    print("\n─── Classification report ────────────────────────────────")
    print(classification_report(
        all_labels, all_preds,
        target_names=list(CLASS_NAMES.keys())
    ))

    # extract F1 per class
    classes = list(CLASS_NAMES.keys())
    f1s     = [report[cls]["f1-score"] for cls in classes]
    colors  = ["#E24B4A" if f < 0.5 else "#EF9F27" if f < 0.7 else "#1D9E75"
               for f in f1s]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars    = ax.bar(classes, f1s, color=colors)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="0.7 threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 scores")
    ax.legend(fontsize=8)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{f1:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("f1_scores.png", dpi=150)
    print("Saved f1_scores.png")
    plt.show()

    return report

# ── ROC CURVES ────────────────────────────────────────────────────────────────

def plot_roc_curves(all_probs, all_labels):
    """
    ROC curve: plots true positive rate vs false positive rate at every threshold.
    AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing.
    One curve per class — each class is treated as binary (this class vs all others).
    """
    # binarize labels: e.g. if true label is 1 (mel),
    # binarized = [0, 1, 0, 0, 0, 0, 0]
    classes        = list(range(NUM_CLASSES))
    labels_bin     = label_binarize(all_labels, classes=classes)
    class_labels   = list(CLASS_NAMES.keys())

    colors = ["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27",
              "#7F77DD", "#D85A30", "#888780"]

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (cls, color) in enumerate(zip(class_labels, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{cls} (AUC = {roc_auc:.2f})")

    # diagonal = random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC = 0.50)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — one vs rest")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("Saved roc_curves.png")
    plt.show()

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def print_summary(report, all_preds, all_labels):
    print("\n─── Summary ──────────────────────────────────────────────")
    correct = (all_preds == all_labels).sum()
    total   = len(all_labels)
    print(f"Overall accuracy : {correct/total*100:.2f}%  ({correct}/{total})")
    print(f"Macro F1         : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1      : {report['weighted avg']['f1-score']:.4f}")
    print()
    print("Per-class breakdown:")
    print(f"  {'Class':<8} {'Full name':<25} {'F1':>6}  {'Precision':>10}  {'Recall':>8}")
    print("  " + "─" * 62)
    for cls in CLASS_NAMES.keys():
        r = report[cls]
        flag = " ← needs improvement" if r["f1-score"] < 0.5 else ""
        print(f"  {cls:<8} {CLASS_NAMES[cls]:<25} {r['f1-score']:>6.2f}  "
              f"{r['precision']:>10.2f}  {r['recall']:>8.2f}{flag}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load model
    model = load_model()
    print(f"Loaded {CHECKPOINT}\n")

    # build val dataloader
    meta_path   = META_PATH if os.path.exists(META_PATH) else META_PATH + ".csv"
    df          = pd.read_csv(meta_path)
    path_lookup = build_path_lookup(IMG_DIRS)

    _, val_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=RANDOM_SEED
    )

    val_ds     = HAM10000Dataset(val_df, path_lookup, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # get all predictions
    print("Running inference on full val set...")
    all_preds, all_labels, all_probs = get_predictions(model, val_loader)

    # plots
    plot_confusion_matrix(all_preds, all_labels)
    report = plot_f1_scores(all_preds, all_labels)
    plot_roc_curves(all_probs, all_labels)
    print_summary(report, all_preds, all_labels)

    print("\nDay 5 complete! Saved: confusion_matrix.png, f1_scores.png, roc_curves.png")