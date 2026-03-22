"""
improved_presidential.py
========================
Author improvement over the original course implementation.

WHAT I ADDED vs original:
  1. Fine-tuning option   — option to unfreeze last VGG16 layers
  2. Early stopping       — stops training when model stops improving
  3. Training curve plot  — saves loss/accuracy graph as PNG
  4. Model save & load    — save best model, reload without retraining
  5. Prediction with confidence — not just "Bo / Not Bo" but HOW sure
  6. Layer freezing report— prints which layers are frozen vs trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = "data/presidential_doggy_door"
MODEL_SAVE_PATH = "best_presidential_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.85   # How confident before letting Bo in

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def get_dataloaders():
    train_data = ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    valid_data = ImageFolder(os.path.join(DATA_DIR, "valid"), transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Classes: {train_data.classes}")
    print(f"  Train samples: {len(train_data)}  |  Valid samples: {len(valid_data)}")

    return train_loader, valid_loader, len(train_data), len(valid_data)


# ─────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────
def build_model(fine_tune_last_layers=False):
    """
    Load VGG16 and attach a custom classifier for Bo detection.

    MY IMPROVEMENT: fine_tune_last_layers option.
    Original: ALL layers frozen.
    Improved: optionally unfreeze last 2 conv layers for fine-tuning.

    Why this matters:
      - Fully frozen = fast training, but less specialized
      - Fine-tuned last layers = slower, but model adapts more to Bo's features
      - Too many unfrozen layers + small data = overfitting (bad)
    """
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)

    # Step 1: Freeze everything first
    model.requires_grad_(False)

    # Step 2 (MY IMPROVEMENT): Optionally unfreeze last 2 conv layers
    if fine_tune_last_layers:
        for param in model.features[24:].parameters():
            param.requires_grad = True
        print("  ⚙️  Fine-tuning: last 2 conv layers UNFROZEN")
    else:
        print("  ❄️  All pretrained layers FROZEN (feature extraction only)")

    # Step 3: Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),        # Added dropout to reduce overfitting
        nn.Linear(1024, 256),   # Extra layer vs original
        nn.ReLU(),
        nn.Linear(256, 2)       # 2 classes: Bo or Not Bo
    )

    print_layer_status(model)
    return model


def print_layer_status(model):
    """
    MY IMPROVEMENT: Print which layers are trainable vs frozen.
    Helps understand exactly what's happening during training.
    """
    print("\n  📋 LAYER STATUS:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"     Trainable params: {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"     Frozen params:    {frozen:,}  ({frozen/total*100:.1f}%)")
    print(f"     Total params:     {total:,}\n")


# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    MY IMPROVEMENT: Stop training when validation loss stops improving.

    Original code: always runs all epochs even if model stopped learning.
    This improvement: saves time + prevents overfitting automatically.
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def check(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True   # Improved → save model
        else:
            self.counter += 1
            print(f"     ⏳ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                print("     🛑 Early stopping triggered!")
            return False  # Did not improve


# ─────────────────────────────────────────────
# TRAINING & VALIDATION
# ─────────────────────────────────────────────
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train_one_epoch(model, loader, N, optimizer, loss_fn):
    model.train()
    total_loss, total_acc = 0, 0

    for x, y in loader:
        output = model(x)
        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += get_batch_accuracy(output, y, N)

    return total_loss, total_acc


def validate_one_epoch(model, loader, N, loss_fn):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            total_loss += loss_fn(output, y).item()
            total_acc += get_batch_accuracy(output, y, N)

    return total_loss, total_acc


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(fine_tune=False):
    print("\n" + "="*60)
    print("  🏛️  PRESIDENTIAL DOGGY DOOR — TRAINING")
    print("="*60)

    # Load data
    train_loader, valid_loader, train_N, valid_N = get_dataloaders()

    # Build model
    model = build_model(fine_tune_last_layers=fine_tune)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LEARNING_RATE)

    # MY IMPROVEMENT: Early stopping
    early_stop = EarlyStopping(patience=5)

    # Track history for plotting
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  Training for up to {MAX_EPOCHS} epochs...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, train_N, optimizer, loss_fn)
        v_loss, v_acc = validate_one_epoch(model, valid_loader, valid_N, loss_fn)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")

        # MY IMPROVEMENT: Save best model + early stopping
        improved = early_stop.check(v_loss)
        if improved:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"     💾 Best model saved! (val_loss: {v_loss:.4f})")

        if early_stop.should_stop:
            print(f"\n  Stopped at epoch {epoch} (best val_loss: {early_stop.best_loss:.4f})")
            break

    # MY IMPROVEMENT: Plot training curves
    plot_training_curves(history)
    return model, history


# ─────────────────────────────────────────────
# TRAINING CURVE PLOT
# ─────────────────────────────────────────────
def plot_training_curves(history):
    """
    MY IMPROVEMENT: Visualize training progress.
    Original code: no visualization at all.
    This shows: whether model is overfitting, when it stopped improving.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Presidential Doggy Door — Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Training curve saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# PREDICTION WITH CONFIDENCE
# ─────────────────────────────────────────────
def load_saved_model():
    """MY IMPROVEMENT: Load saved model without retraining."""
    model = build_model(fine_tune_last_layers=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print(f"  ✅ Model loaded from {MODEL_SAVE_PATH}")
    return model


def predict_bo(model, image_path, class_names=["bo", "not_bo"]):
    """
    MY IMPROVEMENT: Confidence-based prediction.
    Original: just outputs class index.
    Improved: outputs class name + confidence + door decision.
    """
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ⚠️  Could not load image: {e}")
        return

    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = probs.max(dim=1)

    label = class_names[predicted.item()]
    conf = confidence.item()

    print(f"\n  📸 Image: {os.path.basename(image_path)}")
    print(f"  🧠 Prediction: {label.upper()}  ({conf*100:.1f}% confident)")

    if label == "bo" and conf >= CONFIDENCE_THRESHOLD:
        print("  🏛️  Decision: ✅ DOOR OPENS — Welcome, Bo!")
    elif label == "bo" and conf < CONFIDENCE_THRESHOLD:
        print(f"  🏛️  Decision: ❌ DOOR CLOSED — Looks like Bo but not confident enough ({conf*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}%)")
    else:
        print("  🏛️  Decision: ❌ ACCESS DENIED — Not Bo!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  OPTION A: Train with all layers frozen (original approach)")
    print("  OPTION B: Train with fine-tuning (my improvement)")
    print("="*60)

    # Change fine_tune=True to try fine-tuning
    model, history = train_model(fine_tune=False)

    # Test prediction on a saved image
    # predict_bo(model, "images/happy_dog.jpg")
