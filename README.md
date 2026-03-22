# 🐶 Transfer Learning Doggy Door AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA_DLI-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

<br/>

> **An intelligent computer vision system** using pretrained **VGG16** to automate a smart doggy door — extended with **Transfer Learning** to recognize a *specific* dog.  
> Built on top of the NVIDIA DLI course, then **independently improved** with confidence scoring, early stopping, model comparison, and training visualizations.

<br/>

```
🐕 Dog detected         →  ✅ Door Opens
🐱 Cat detected         →  ❌ Door Stays Closed
🐻 Bear detected        →  ❌ Door Stays Closed
🐾 Bo (First Dog) only  →  ✅ VIP Entry Granted
```

</div>

---

## 🎬 Video Walkthrough

<div align="center">

[![YouTube Video](https://img.shields.io/badge/Watch%20on-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@YourChannelHere)

> 📺 **I made a full video explaining this project** — original code walkthrough + all 3 improvements explained line by line.  
> Watch it to understand exactly what I changed and **why** each improvement matters.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [My Improvements](#-my-improvements-vs-original)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Visual Diagrams](#-visual-diagrams)
- [Part 1 — Doggy Door](#-part-1--doggy-door-pretrained-vgg16)
- [Part 2 — Presidential Doggy Door](#-part-2--presidential-doggy-door-transfer-learning)
- [Utils & Helper Functions](#-utils--helper-functions)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Key Concepts](#-key-concepts)

---

## 🚀 Project Overview

| Stage | File | Task | Approach |
|-------|------|------|----------|
| 🐶 Part 1 | `05a_doggy_door.ipynb` | Detect any dog → open door | Pretrained VGG16 (ImageNet) |
| 🏛️ Part 2 | `05b_presidential_doggy_door.ipynb` | Detect only Bo → secure entry | Transfer Learning on VGG16 |
| 🔥 Improved 1 | `improved_doggy_door.py` | Confidence + batch + model compare | **My improvements** |
| 🔥 Improved 2 | `improved_presidential.py` | Fine-tuning + early stopping + plots | **My improvements** |

---

## 🔥 My Improvements vs Original

> The original course notebooks gave me the foundation. I then extended both implementations independently.

### `improved_doggy_door.py`

| Feature | Original Course | My Improvement |
|---------|----------------|----------------|
| Decision | Dog → Open | Dog **+ confidence ≥ 80%** → Open |
| Processing | One image at a time | **Batch processing** — any number of images |
| Error handling | Crashes on bad image | **Safe loader** with try/except |
| Output | Print statement | **Formatted summary report** |
| Models tested | VGG16 only | **VGG16 vs ResNet50 comparison** |

**Confidence + Threshold Logic I added:**
```python
# MY IMPROVEMENT: Two-layer decision
# Original just checked: is it a dog?
# I also check: how SURE is the model?

probabilities = F.softmax(outputs, dim=1)
confidence, predicted_class = probabilities.max(dim=1)

if is_dog(predicted_class) and confidence >= 0.80:
    print(f"✅ Door OPENS  ({confidence*100:.1f}% confident)")
elif is_dog(predicted_class):
    print(f"❌ Uncertain  ({confidence*100:.1f}% — below 80% threshold)")
else:
    print("❌ Not a dog — Door CLOSED")
```

### `improved_presidential.py`

| Feature | Original Course | My Improvement |
|---------|----------------|----------------|
| Layer freezing | All layers frozen | **Option to fine-tune last layers** |
| Training | Fixed epochs | **Early stopping** (stops when not improving) |
| Visualization | None | **Training curve plot** saved as PNG |
| Model saving | Not implemented | **Save best model**, reload without retraining |
| Prediction | Class index only | **Class name + confidence + door decision** |
| Architecture | 1 hidden layer | **2 hidden layers + Dropout** to reduce overfitting |

**Early Stopping I built:**
```python
class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Prevents overfitting on the small Bo dataset.
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def check(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True   # Save model
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
```

**Fine-tuning option I added:**
```python
# MY IMPROVEMENT: Selectively unfreeze last conv layers
# Original: model.requires_grad_(False)  — everything frozen
# Mine: gives option to fine-tune the last 2 layers

model.requires_grad_(False)   # Freeze everything first

if fine_tune_last_layers:
    for param in model.features[24:].parameters():
        param.requires_grad = True   # Unfreeze only last 2 conv layers
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Core language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ Torchvision | Pretrained VGG16 + ResNet50 |
| 📓 Jupyter Notebook | Development & exploration |
| 📊 Matplotlib | Training curve visualization |
| 🎮 NVIDIA DLI | Original course platform |

---

## 🧠 How It Works

### Architecture Flow

```
Input Image (224×224 RGB)
         │
         ▼
┌─────────────────────────┐
│   VGG16 Pretrained      │  ← Trained on 1.2M ImageNet images
│   (Feature Extractor)   │    Learns: edges → textures → objects
│   [FROZEN]              │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Custom Classifier     │  ← Built by me, trained on Bo photos
│   Linear → ReLU → Drop  │
│   Linear → ReLU         │
│   Linear(2)             │
└─────────────────────────┘
         │
         ▼
    Softmax Probabilities
         │
         ▼
  Confidence Threshold Check  ← MY IMPROVEMENT
         │
         ▼
   OPEN DOOR / DENY ACCESS
```

---

## 🖼️ Visual Diagrams

### Diagram 1 — Transfer Learning Architecture (Feature Spectrum)

> This diagram shows the full pipeline I used. Left side = frozen pretrained layers (general knowledge). Right side = my custom trainable layers (specific to the task).

![Transfer Learning Architecture](diagrams/transfer_learning_architecture.png)

**What this shows:**
- **Conv Stage 1** learns low-level features — edges and curves
- **Conv Stage 2** learns mid-level features — patterns and textures  
- **Conv Stage 3** learns high-level features — simple object parts
- **Conv Stage 4** learns complex deep features — specific object shapes
- All 4 stages are **frozen** (locked 🔒) — we reuse VGG16's pretrained knowledge
- The **Custom Classification Head** on the right is the only part we train — this is where Bo detection happens
- The **Softmax output** gives us confidence scores — which is Improvement #1 I added

---

### Diagram 2 — CNN Layer-by-Layer (How Convolutions Work)

> This diagram shows exactly how an image travels through the network — from raw pixels to a final prediction with confidence score.

![CNN Transfer Learning Diagram](diagrams/cnn_transfer_learning.png)

**What this shows:**
- Image goes in as raw pixels (e.g. 1 × 28 × 28)
- Kernels (small filters) slide across the image to detect features — this is `nn.Conv2d` in PyTorch
- Max Pooling shrinks the image while keeping important features — this is `nn.MaxPool2d`
- After all conv layers, the image is **Flattened** into a 1D vector — a list of numbers
- That vector goes into **Fully Connected (Linear) layers** — this is `nn.Linear` in PyTorch
- Final output = prediction with confidence score from Softmax

---

### What VGG16 Learned Layer by Layer

```
Conv Layer 1-3  →  Edges, lines, gradients
Conv Layer 4-7  →  Textures, patterns
Conv Layer 8-13 →  Shapes, object parts (fur, paws, eyes)
Conv Layer 14-16→  Full objects (dogs, cats, etc.)

We KEEP all of this (frozen) and only train the final decision layers.
```

---

## 🐶 Part 1 — Doggy Door (Pretrained VGG16)

### Load the Pretrained Model

```python
from torchvision.models import vgg16, VGG16_Weights

weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)
model.eval()
```

### Image Preprocessing

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Door Decision Logic

```python
# ImageNet dog classes = indices 151 to 268
outputs = model(image)
_, predicted_idx = torch.max(outputs, 1)

if 151 <= predicted_idx.item() <= 268:
    print("🐕 Dog → Door Opens")
else:
    print("🚫 Not a dog → Door Closed")
```

---

## 🏛️ Part 2 — Presidential Doggy Door (Transfer Learning)

### The Problem

```
VGG16 sees any dog → "It's a dog"  ✅
VGG16 sees Bo     → "It's a dog"  ✅  ← Can't tell the difference!

We need: Bo → "It's Bo" ✅ | Other dog → "Not Bo" ❌
```

### Transfer Learning Strategy

```
┌──────────────────────────────────────────────────────┐
│  VGG16 Pretrained Layers (FROZEN — reused as-is)     │
│                                                      │
│  features[0..23]  → General visual knowledge         │
│  features[24..]   → (optionally fine-tuned)          │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Custom Classifier (TRAINABLE — trained on Bo)       │
│                                                      │
│  Linear(25088 → 1024) → ReLU → Dropout(0.5)          │
│  Linear(1024 → 256)   → ReLU                         │
│  Linear(256 → 2)      → [Bo, Not Bo]                 │
└──────────────────────────────────────────────────────┘
```

### Build the Model

```python
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.requires_grad_(False)   # Freeze pretrained layers

model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)
```

### Training

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```

---

## ⚙️ Utils & Helper Functions

### MyConvBlock

```python
class MyConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → Dropout → MaxPool"""
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )
    def forward(self, x):
        return self.model(x)
```

### train / validate

```python
def train(model, train_loader, train_N, random_trans, optimizer, loss_function):
    model.train()
    for x, y in train_loader:
        output = model(random_trans(x))
        optimizer.zero_grad()
        loss_function(output, y).backward()
        optimizer.step()

def validate(model, valid_loader, valid_N, loss_function):
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
```

---

## 📊 Results

### Part 1 — General Doggy Door

| Input Image | VGG16 Class | Dog? | Confidence | Door |
|-------------|------------|------|------------|------|
| happy_dog.jpg | Golden Retriever (207) | ✅ | High | 🔓 Opens |
| sleepy_cat.jpg | Tabby Cat (281) | ❌ | High | 🔒 Closed |
| brown_bear.jpg | Brown Bear (294) | ❌ | High | 🔒 Closed |

### Part 2 — Presidential Doggy Door

| Input | Prediction | Confidence | Access |
|-------|-----------|------------|--------|
| Bo photo | Bo ✅ | > 85% | 🔓 Granted |
| Other dog | Not Bo ❌ | — | 🔒 Denied |
| Cat / Bear | Not Bo ❌ | — | 🔒 Denied |

---

## 📁 Project Structure

```
transfer-learning-doggy-door-ai/
│
├── 📓 05a_doggy_door.ipynb                 ← Original course notebook
├── 📓 05b_presidential_doggy_door.ipynb    ← Original course notebook
├── 🐍 utils.py                             ← Original course utilities
│
├── 🔥 improved_doggy_door.py               ← My improvements (Part 1)
├── 🔥 improved_presidential.py             ← My improvements (Part 2)
│
├── 📂 data/                                ← Bo training images
├── 📂 images/                              ← Test images
│
├── 📊 training_curves.png                  ← Auto-generated by improved code
└── 📄 README.md
```

---

## 🧠 Key Concepts

### Transfer Learning

```
❌ Train from scratch:
   Need thousands of Bo images → weeks of training → likely overfits

✅ Transfer Learning:
   Reuse VGG16's 138M parameters of visual knowledge
   Train only new layers → works with ~30 images
```

### Freezing vs Fine-tuning

```
Frozen     →  weights don't update  →  safe for tiny datasets
Fine-tuned →  weights slowly adapt  →  risky if too much data is too small
                                       (I explore both in improved_presidential.py)
```

### Feature Hierarchy

```
Layer 1-3  (Early)   →  Edges, colors
Layer 4-7  (Middle)  →  Textures, shapes
Layer 8-16 (Deep)    →  Full objects

We reuse early/middle (general) → replace deep (make it specific to Bo)
```

### ⚠️ Data Bias

Transfer learning can inherit biases from the original training data. If ImageNet underrepresents certain visual environments or animal types, our model may perform poorly in those cases. Always test across diverse inputs.

---

## 🏆 What I Learned

- ✅ Loading and using a **pretrained CNN (VGG16)** for real-world classification
- ✅ How **ImageNet class indices** map to object categories
- ✅ **Transfer learning** — reusing knowledge from one task for another
- ✅ **Layer freezing** and why it prevents overfitting on small datasets
- ✅ **Confidence thresholding** to make predictions more reliable
- ✅ **Early stopping** to avoid wasting compute and overfitting
- ✅ Why **data bias** is a real concern in deployed AI systems

---

## 🙏 Acknowledgements

- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/) — course foundation and dataset
- [VGG16 Paper](https://arxiv.org/abs/1409.1556) — Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)
- [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)

---

<div align="center">

Built on NVIDIA DLI Foundation · Extended & Improved Independently · PyTorch 🔥

</div>
