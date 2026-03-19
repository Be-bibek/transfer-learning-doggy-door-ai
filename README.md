# 🐶 Transfer Learning Doggy Door AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA_DLI-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

<br/>

> **An intelligent computer vision system** that uses a pretrained **VGG16** model to automate a smart doggy door — and extends it with **Transfer Learning** to recognize a *specific* dog.

<br/>

```
🐕 Dog detected  →  ✅ Door Opens
🐱 Cat detected  →  ❌ Door Stays Closed
🐻 Bear detected →  ❌ Door Stays Closed
🐾 Bo (First Dog) detected → ✅ VIP Entry Granted
```

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Part 1 — Doggy Door (Pretrained VGG16)](#-part-1--doggy-door-pretrained-vgg16)
- [Part 2 — Presidential Doggy Door (Transfer Learning)](#-part-2--presidential-doggy-door-transfer-learning)
- [Utils & Helper Functions](#-utils--helper-functions)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Key Concepts](#-key-concepts)
- [Dataset](#-dataset)

---

## 🚀 Project Overview

This project is split into **two real-world AI applications**, both powered by deep learning:

| Stage | Notebook | Task | Approach |
|-------|----------|------|----------|
| 🐶 Part 1 | `05a_doggy_door.ipynb` | Detect any dog → open door | Pretrained VGG16 (ImageNet) |
| 🏛️ Part 2 | `05b_presidential_doggy_door.ipynb` | Detect *only Bo* → secure entry | Transfer Learning on VGG16 |

The key insight here:
> **Part 1** uses a model that already knows what a dog looks like.  
> **Part 2** reuses that knowledge but specializes it — so only the *First Dog of the USA (Bo)* can enter.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Core language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ Torchvision | Pretrained VGG16 model |
| 📓 Jupyter Notebook | Development environment |
| 🎮 NVIDIA DLI | Training platform |
| 📊 Matplotlib | Visualization |

---

## 🧠 How It Works

### 🔷 Overall Architecture

```
Input Image (224×224 RGB)
         │
         ▼
┌─────────────────────────┐
│   VGG16 Pretrained      │  ← Trained on 1.2M ImageNet images
│   (Feature Extractor)   │    Knows: edges → shapes → objects
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Custom Classifier     │  ← We build this on top
│   (Our New Layers)      │    Trained on our small dataset
└─────────────────────────┘
         │
         ▼
    Prediction
  (Dog / Not Dog / Bo)
```

### 🔷 Why VGG16?

VGG16 won the **2014 ImageNet Challenge** — trained on millions of labeled images across 1000 categories. Instead of training from scratch, we *reuse* its powerful feature extraction ability.

```
Convolutional Layers (what VGG16 learned):

Layer 1:  Detects → Edges, Lines
Layer 3:  Detects → Textures, Patterns
Layer 7:  Detects → Shapes, Contours
Layer 13: Detects → Object Parts (fur, eyes, paws)
Layer 16: Detects → Full Objects (dogs, cats, bears)
```

---

## 🐶 Part 1 — Doggy Door (Pretrained VGG16)

### Load the Pretrained Model

```python
from torchvision.models import vgg16, VGG16_Weights

# Load VGG16 pretrained on ImageNet
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)
```

### Image Preprocessing

```python
from torchvision import transforms

# VGG16 expects 224×224 normalized images
IMG_WIDTH, IMG_HEIGHT = 224, 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

### Doggy Door Logic

```python
import torch

def run_doggy_door(image_path):
    # Preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # Run through VGG16
    with torch.no_grad():
        outputs = model(image)

    # VGG16 returns 1000 class scores
    # ImageNet dog classes are in range 151-268
    _, predicted_idx = torch.max(outputs, 1)

    if 151 <= predicted_idx.item() <= 268:
        print("🐕 Dog detected → ✅ Door Opens!")
    else:
        print("🚫 Not a dog → ❌ Door Stays Closed")
```

### Prediction Results

```
📸 happy_dog.jpg   →  Class: 207 (Golden Retriever)  →  ✅ Door OPENS
📸 sleepy_cat.jpg  →  Class: 281 (Tabby Cat)         →  ❌ Door CLOSED
📸 brown_bear.jpg  →  Class: 294 (Brown Bear)         →  ❌ Door CLOSED
```

---

## 🏛️ Part 2 — Presidential Doggy Door (Transfer Learning)

### The Problem

VGG16 can detect *any* dog — but the White House only wants **Bo** (the Portuguese Water Dog, First Dog 2009–2017) to enter.

```
❌ Problem:   VGG16 → "It's a dog"  (all dogs look the same to it)
✅ Solution:  Transfer Learning → "It's specifically Bo"
```

### Transfer Learning Strategy

```
Old VGG16 Model (Frozen)         New Custom Layers (Trainable)
─────────────────────────         ──────────────────────────────
Input → Conv1 → Conv2 ...   →    Linear(25088, 1024)
(These learned general             → ReLU()
 features from ImageNet,           → Linear(1024, 2)
 we reuse them as-is)              → Softmax
                                  Output: [Bo, Not Bo]
```

### Code

```python
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# Step 1: Load pretrained VGG16
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)

# Step 2: Freeze ALL pretrained layers
# (We don't want to destroy what VGG16 already learned)
model.requires_grad_(False)

# Step 3: Replace the classifier with our own
# (This is the only part that will train on Bo's photos)
model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),   # 25088 = VGG16 feature size
    nn.ReLU(),
    nn.Linear(1024, 2)        # 2 classes: Bo or Not Bo
)

print("Model ready for transfer learning!")
```

### Why Freeze the Old Layers?

```
If we DON'T freeze:
  → All 138 million weights update
  → Only have ~30 images of Bo
  → Model OVERFITS badly ❌

If we DO freeze:
  → Only ~26 million new weights update
  → Small dataset is enough
  → Model generalizes well ✅
```

### Training Setup

```python
from torch.optim import Adam

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
```

### Training Loop

```python
epochs = 10

for epoch in range(epochs):
    # Training phase
    train(model, train_loader, train_N, random_trans, optimizer, loss_function)

    # Validation phase
    validate(model, valid_loader, valid_N, loss_function)
```

---

## ⚙️ Utils & Helper Functions

The `utils.py` file contains core reusable components:

### MyConvBlock — Custom Conv Layer

```python
import torch
import torch.nn as nn

class MyConvBlock(nn.Module):
    """
    A reusable convolutional block:
    Conv2D → BatchNorm → ReLU → Dropout → MaxPool
    """
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)
```

### get_batch_accuracy — Accuracy Calculator

```python
def get_batch_accuracy(output, y, N):
    """
    Calculates accuracy for one batch.
    output: model predictions
    y: true labels
    N: number of samples
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N
```

### train — Training Function

```python
def train(model, train_loader, train_N, random_trans, optimizer, loss_function):
    loss = 0
    accuracy = 0

    model.train()  # Set to training mode
    for x, y in train_loader:
        output = model(random_trans(x))     # Forward pass with augmentation
        optimizer.zero_grad()               # Clear old gradients
        batch_loss = loss_function(output, y)
        batch_loss.backward()               # Backpropagation
        optimizer.step()                    # Update weights

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)

    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
```

### validate — Validation Function

```python
def validate(model, valid_loader, valid_N, loss_function):
    loss = 0
    accuracy = 0

    model.eval()  # Set to evaluation mode (no dropout)
    with torch.no_grad():   # No gradient calculation needed
        for x, y in valid_loader:
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)

    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
```

---

## 📊 Results

### Part 1 — General Doggy Door

| Input Image | VGG16 Prediction | Dog? | Door Action |
|-------------|-----------------|------|-------------|
| 🐕 happy_dog.jpg | Golden Retriever (207) | ✅ Yes | 🔓 Opens |
| 🐱 sleepy_cat.jpg | Tabby Cat (281) | ❌ No | 🔒 Closed |
| 🐻 brown_bear.jpg | Brown Bear (294) | ❌ No | 🔒 Closed |

### Part 2 — Presidential Doggy Door

| Input | Transfer Model Says | Access |
|-------|-------------------|--------|
| Bo (Portuguese Water Dog) | ✅ It's Bo! | 🔓 Granted |
| Other dog | ❌ Not Bo | 🔒 Denied |
| Cat / Bear / etc | ❌ Not Bo | 🔒 Denied |

---

## 📁 Project Structure

```
transfer-learning-doggy-door-ai/
│
├── 📓 05a_doggy_door.ipynb            # Part 1: VGG16 pretrained doggy door
├── 📓 05b_presidential_doggy_door.ipynb  # Part 2: Transfer learning for Bo
├── 🐍 utils.py                        # Helper: train, validate, conv block
│
├── 📂 data/                           # Training data for Bo detection
│   ├── presidential_doggy_door/
│   │   ├── train/
│   │   └── valid/
│
├── 📂 images/                         # Test images
│   ├── happy_dog.jpg
│   ├── sleepy_cat.jpg
│   └── brown_bear.jpg
│
└── 📄 README.md
```

---

## 🧠 Key Concepts

### Transfer Learning (Core Idea)

```
❌ WITHOUT Transfer Learning:
   Need thousands of Bo photos
   Train for hours/days
   Risk of overfitting

✅ WITH Transfer Learning:
   Just ~30 photos of Bo
   Train in minutes
   Works great on small data
```

### Feature Hierarchy in CNNs

```
Layer 1-3  (Early)   → Edges, Colors, Gradients
Layer 4-7  (Middle)  → Textures, Patterns, Shapes  
Layer 8-13 (Deep)    → Object Parts (eyes, paws, fur)
Layer 14-16 (Final)  → Full Objects, Scene Context

💡 We KEEP early/middle layers (general)
💡 We REPLACE final layers (make them specific to Bo)
```

### Freezing vs Fine-tuning

```
Frozen Layers:    weights DON'T change during training
                  → preserves pretrained knowledge
                  → safe for small datasets

Trainable Layers: weights DO change during training
                  → specializes the model
                  → risky if too many params + small data
```

### ⚠️ Data Bias (Important Note)

Transfer learning can carry over biases from the original model. If ImageNet was biased in how it labeled certain animals or environments, our new model may inherit those biases. Always evaluate your model's performance across diverse inputs.

---

## 📦 Dataset

- **Part 1:** Uses VGG16 pretrained weights on **ImageNet** (1.2M images, 1000 classes) — no extra data needed.
- **Part 2:** Small dataset of **Bo** photos provided by NVIDIA DLI course.

> Dataset courtesy of [NVIDIA Deep Learning Institute (DLI)](https://www.nvidia.com/en-us/training/)

---

## 🏆 What I Learned

- ✅ How to load and use a **pretrained CNN model** (VGG16)
- ✅ How **ImageNet class indices** map to real-world objects
- ✅ The concept of **Transfer Learning** and why it works
- ✅ How to **freeze layers** and build custom classifier heads
- ✅ Why **data bias** matters in AI systems
- ✅ How to build a **real-world AI pipeline** end-to-end

---

## 🙏 Acknowledgements

- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/) — course material and dataset
- [VGG16 Paper](https://arxiv.org/abs/1409.1556) — Very Deep Convolutional Networks for Large-Scale Image Recognition
- [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/) — framework

---

<div align="center">

Made with 🧠 + 🔥 PyTorch | NVIDIA DLI Deep Learning Course

</div>
