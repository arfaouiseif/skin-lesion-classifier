import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR    = r"C:\Users\arfaoui\Downloads\dataverse_files"
IMG_DIRS    = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
META_PATH   = os.path.join(DATA_DIR, "HAM10000_metadata")
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_WORKERS = 4
RANDOM_SEED = 42

# ─── LABEL MAP ────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    "nv":    "Melanocytic nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign keratosis",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "vasc":  "Vascular lesion",
    "df":    "Dermatofibroma",
}
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES.keys())}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

# ─── BUILD IMAGE PATH LOOKUP ──────────────────────────────────────────────────

def build_path_lookup(img_dirs):
    lookup = {}
    for folder in img_dirs:
        for fname in os.listdir(folder):
            img_id = os.path.splitext(fname)[0]  # strip .jpg
            lookup[img_id] = os.path.join(folder, fname)
    return lookup

# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(self, df, path_lookup, transform=None):
        self.df           = df.reset_index(drop=True)
        self.path_lookup  = path_lookup
        self.transform    = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.path_lookup[row["image_id"]]
        image    = Image.open(img_path).convert("RGB")
        label    = CLASS_TO_IDX[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load metadata
    # try both with and without .csv extension
    meta_path = META_PATH if os.path.exists(META_PATH) else META_PATH + ".csv"
    df = pd.read_csv(meta_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # 2. Class distribution
    print("─── Class distribution ───────────────────────")
    dist = df["dx"].value_counts()
    for cls, count in dist.items():
        pct  = count / len(df) * 100
        name = CLASS_NAMES.get(cls, cls)
        print(f"  {cls:6s} ({name:25s}): {count:5d}  ({pct:.1f}%)")
    print()

    # 3. Compute class weights for imbalanced training
    counts       = df["dx"].map(dist).values
    class_counts = np.array([dist[cls] for cls in CLASS_NAMES.keys()])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    print("─── Class weights (for loss function) ────────")
    for cls, w in zip(CLASS_NAMES.keys(), class_weights):
        print(f"  {cls:6s}: {w:.4f}")
    print()

    # 4. Build image path lookup
    path_lookup = build_path_lookup(IMG_DIRS)
    print(f"Found {len(path_lookup)} images on disk\n")

    # 5. Train / val split (80/20, stratified)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["dx"],
        random_state=RANDOM_SEED,
    )
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples\n")

    # 6. Create datasets and dataloaders
    train_dataset = HAM10000Dataset(train_df, path_lookup, transform=train_transforms)
    val_dataset   = HAM10000Dataset(val_df,   path_lookup, transform=val_transforms)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 7. Sanity check — load one batch
    print("─── Sanity check ─────────────────────────────")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape : {images.shape}")   # [64, 3, 224, 224]
    print(f"Batch label shape : {labels.shape}")   # [64]
    print(f"Label sample      : {[IDX_TO_CLASS[l.item()] for l in labels[:8]]}")
    print()

    # 8. Plot class distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ["#E24B4A" if cls in ("mel", "bcc") else
               "#EF9F27" if cls == "akiec" else
               "#378ADD" for cls in dist.index]
    ax.bar([CLASS_NAMES.get(c, c) for c in dist.index], dist.values, color=colors)
    ax.set_title("HAM10000 class distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    print("Saved class_distribution.png")

    # 9. Plot sample images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        row      = df.sample(1).iloc[0]
        img_path = path_lookup[row["image_id"]]
        img      = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{row['dx']} — {CLASS_NAMES[row['dx']][:12]}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Sample images from HAM10000", fontsize=12)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    print("Saved sample_images.png")

    print("\nDay 1 complete! Pipeline is working.")
    print(f"GPU available: {torch.cuda.is_available()}")

    return train_loader, val_loader, class_weights

if __name__ == "__main__":
    train_loader, val_loader, class_weights = main()