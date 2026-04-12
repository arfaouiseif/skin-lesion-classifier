# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Check GPU and install missing library
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess
subprocess.run(["pip", "install", "grad-cam", "-q"])

import torch
print(f"PyTorch version : {torch.__version__}")
print(f"GPU available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ═══════════════════════════════════════════════════════════════════════════════

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

print("All imports OK")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Config
# ═══════════════════════════════════════════════════════════════════════════════

# ── paths (Kaggle mounts datasets here automatically) ─────────────────────────
DATA_DIR  = "/kaggle/input/isic-2019"
IMG_DIR   = os.path.join(DATA_DIR, "ISIC_2019_Training_Input",
                         "ISIC_2019_Training_Input")
META_PATH = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")

# ── training ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
BATCH_SIZE   = 64
NUM_EPOCHS   = 25
LR_HEAD      = 1e-4    # phase 1: frozen backbone, train head only
LR_FINETUNE  = 1e-5    # phase 2: unfreeze all, fine-tune
RANDOM_SEED  = 42
NUM_WORKERS  = 2        # Kaggle recommends 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH    = "/kaggle/working/best_model_isic2019.pth"

print(f"Device     : {DEVICE}")
print(f"Image dir  : {IMG_DIR}")
print(f"Meta path  : {META_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Explore dataset
# ═══════════════════════════════════════════════════════════════════════════════

df_raw = pd.read_csv(META_PATH)
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
df_raw.head()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Build label map from ground truth
# ═══════════════════════════════════════════════════════════════════════════════

# ISIC 2019 ground truth is one-hot encoded — convert to single label column
label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# keep only rows that have exactly one label (drop "UNK" unknowns)
df = df_raw[df_raw[label_cols].sum(axis=1) == 1].copy()
df["dx"] = df[label_cols].idxmax(axis=1).str.lower()
df["image_path"] = df["image"].apply(
    lambda x: os.path.join(IMG_DIR, x + ".jpg")
)

# verify all images exist
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Total usable samples: {len(df)}")

# class distribution
dist = df["dx"].value_counts()
print("\nClass distribution:")
for cls, count in dist.items():
    print(f"  {cls:6s}: {count:5d}  ({count/len(df)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Label encoding + class weights
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {
    "nv":   "Melanocytic nevi",
    "mel":  "Melanoma",
    "bkl":  "Benign keratosis",
    "bcc":  "Basal cell carcinoma",
    "ak":   "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df":   "Dermatofibroma",
    "scc":  "Squamous cell carcinoma",
}
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES.keys())}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

# align df to our class order
class_counts  = np.array([dist.get(cls, 1) for cls in CLASS_NAMES.keys()])
class_weights = torch.tensor(
    1.0 / class_counts / (1.0 / class_counts).sum() * NUM_CLASSES,
    dtype=torch.float32
).to(DEVICE)

print("Class weights:")
for cls, w in zip(CLASS_NAMES.keys(), class_weights):
    print(f"  {cls:6s}: {w:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Dataset class + transforms
# ═══════════════════════════════════════════════════════════════════════════════

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class ISIC2019Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# train / val split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["dx"], random_state=RANDOM_SEED
)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

train_ds     = ISIC2019Dataset(train_df, transform=train_transforms)
val_ds       = ISIC2019Dataset(val_df,   transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# sanity check
imgs, lbls = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}  labels: {lbls.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Build model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(freeze_backbone=True):
    model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features

    # freeze backbone for phase 1
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # replace head with 8-class output (ISIC 2019 has 8 classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feats, NUM_CLASSES)
    )
    return model.to(DEVICE)

model = build_model(freeze_backbone=True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (phase 1): {trainable:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Train / eval functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            running_loss  += loss.item() * images.size(0)
            correct       += (outputs.argmax(1) == labels).sum().item()
            total         += labels.size(0)
    return running_loss / total, correct / total


def run_training(model, train_loader, val_loader,
                 lr, num_epochs, phase_name):
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    history      = {"train_loss": [], "val_loss": [],
                    "train_acc":  [], "val_acc":  []}
    best_acc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            saved        = " ← best"
        else:
            saved = ""

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}{saved}")

    print(f"\nBest val accuracy ({phase_name}): {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model, history, best_acc


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Phase 1: train head only (frozen backbone)
# ═══════════════════════════════════════════════════════════════════════════════

model, history_p1, best_p1 = run_training(
    model, train_loader, val_loader,
    lr=LR_HEAD, num_epochs=10,
    phase_name="Phase 1 — head only (frozen backbone)"
)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Phase 2: unfreeze all layers and fine-tune
# ═══════════════════════════════════════════════════════════════════════════════

# unfreeze everything
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (phase 2): {trainable:,}")

model, history_p2, best_p2 = run_training(
    model, train_loader, val_loader,
    lr=LR_FINETUNE, num_epochs=NUM_EPOCHS,
    phase_name="Phase 2 — full fine-tuning"
)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Save model + plot curves
# ═══════════════════════════════════════════════════════════════════════════════

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# combine histories
combined = {
    "train_loss": history_p1["train_loss"] + history_p2["train_loss"],
    "val_loss":   history_p1["val_loss"]   + history_p2["val_loss"],
    "train_acc":  history_p1["train_acc"]  + history_p2["train_acc"],
    "val_acc":    history_p1["val_acc"]    + history_p2["val_acc"],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(combined["train_loss"], label="Train loss")
ax1.plot(combined["val_loss"],   label="Val loss")
ax1.axvline(x=10, color="gray", linestyle="--", linewidth=0.8, label="Phase 2 starts")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(combined["train_acc"], label="Train acc")
ax2.plot(combined["val_acc"],   label="Val acc")
ax2.axvline(x=10, color="gray", linestyle="--", linewidth=0.8, label="Phase 2 starts")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves_isic2019.png", dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — Final evaluation on val set
# ═══════════════════════════════════════════════════════════════════════════════

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\n─── Final classification report ──────────────────────────")
print(classification_report(
    all_labels, all_preds,
    target_names=list(CLASS_NAMES.keys())
))

accuracy = (all_preds == all_labels).mean()
print(f"Final val accuracy: {accuracy*100:.2f}%")
print(f"\nModel saved at: {SAVE_PATH}")
print("Download it from the Output tab on the right →")
