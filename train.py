import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
import pandas as pd

# ── re-use everything from Day 1 ──────────────────────────────────────────────
from data import (
    HAM10000Dataset,
    build_path_lookup,
    train_transforms,
    val_transforms,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    DATA_DIR,
    IMG_DIRS,
    META_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
NUM_EPOCHS  = 20
LR          = 1e-4        # learning rate
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "best_model.pth"

print(f"Training on: {DEVICE}")

# ── BUILD DATALOADERS ─────────────────────────────────────────────────────────

def get_dataloaders():
    meta_path = META_PATH if os.path.exists(META_PATH) else META_PATH + ".csv"
    df        = pd.read_csv(meta_path)

    # class weights
    dist         = df["dx"].value_counts()
    class_counts = np.array([dist[cls] for cls in CLASS_NAMES.keys()])
    class_weights = torch.tensor(
        1.0 / class_counts / (1.0 / class_counts).sum() * NUM_CLASSES,
        dtype=torch.float32
    ).to(DEVICE)

    path_lookup = build_path_lookup(IMG_DIRS)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=RANDOM_SEED
    )

    train_ds = HAM10000Dataset(train_df, path_lookup, transform=train_transforms)
    val_ds   = HAM10000Dataset(val_df,   path_lookup, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights

# ── MODEL ─────────────────────────────────────────────────────────────────────

def build_model():
    # load ResNet50 pretrained on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # freeze all layers — we don't want to destroy pretrained features yet
    for param in model.parameters():
        param.requires_grad = False

    # replace the final fully connected layer with our 7-class head
    # ResNet50's fc layer is: Linear(2048 → 1000)
    # we replace it with:     Linear(2048 → 7)
    in_features     = model.fc.in_features          # 2048
    model.fc        = nn.Sequential(
        nn.Dropout(p=0.4),                          # regularization
        nn.Linear(in_features, NUM_CLASSES)         # our 7-class output
    )

    return model.to(DEVICE)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # clear gradients from last step
        outputs = model(images)        # forward pass
        loss    = criterion(outputs, labels)  # compute loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():              # no gradients needed during eval
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total

# ── FULL TRAINING RUN ─────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # only optimize the new fc layer (all others are frozen)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    # reduce LR by 0.5 if val loss doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # save best model
        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, SAVE_PATH)
            saved_str = " ← best saved"
        else:
            saved_str = ""

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
            f"{saved_str}"
        )

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model, history

# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train loss")
    ax1.plot(history["val_loss"],   label="Val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train acc")
    ax2.plot(history["val_acc"],   label="Val acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves1.png", dpi=150)
    print("Saved training_curves.png")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, class_weights = get_dataloaders()
    model                                   = build_model()

    print(f"\nModel head: {model.fc}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    model, history = train(model, train_loader, val_loader, class_weights)
    plot_history(history)

    print("\nDay 2 complete! Model trained and saved to best_model.pth")