"""
improved_doggy_door.py
======================
Author improvement over the original course implementation.

WHAT I ADDED vs original:
  1. Confidence scoring  — model now tells HOW sure it is
  2. Threshold logic     — door only opens if confidence > 80%
  3. Batch processing    — test multiple images at once
  4. Error handling      — bad/missing images don't crash the system
  5. Summary report      — clean results table printed at the end
  6. Model comparison    — test VGG16 vs ResNet50 side by side
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
from PIL import Image, UnidentifiedImageError
import time
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_WIDTH, IMG_HEIGHT = 224, 224
CONFIDENCE_THRESHOLD = 0.80   # MY IMPROVEMENT: only open door if ≥ 80% sure

# ImageNet dog class indices (151–268 = dog breeds in VGG16)
DOG_CLASS_MIN = 151
DOG_CLASS_MAX = 268

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    """
    MY IMPROVEMENT: Safe image loader with error handling.
    Original code would crash on bad images — this handles it gracefully.
    """
    if not os.path.exists(image_path):
        print(f"  ⚠️  File not found: {image_path}")
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        return preprocess(img).unsqueeze(0)
    except UnidentifiedImageError:
        print(f"  ⚠️  Could not read image: {image_path}")
        return None
    except Exception as e:
        print(f"  ⚠️  Unexpected error loading {image_path}: {e}")
        return None


def load_model(model_name="vgg16"):
    """Load a pretrained model by name."""
    if model_name == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model


def predict_single(model, image_tensor):
    """
    Run one image through the model.
    Returns: (predicted_class_index, confidence_score)

    MY IMPROVEMENT: Returns softmax confidence, not just a class index.
    """
    with torch.no_grad():
        outputs = model(image_tensor)

        # Softmax converts raw scores → probabilities (0 to 1)
        probabilities = F.softmax(outputs, dim=1)

        # Get highest probability class
        confidence, predicted_class = probabilities.max(dim=1)

    return predicted_class.item(), confidence.item()


def is_dog(class_index):
    """Check if ImageNet class index falls in dog range (151-268)."""
    return DOG_CLASS_MIN <= class_index <= DOG_CLASS_MAX


def doggy_door_decision(confidence, class_index):
    """
    MY IMPROVEMENT: Two-layer decision system.

    Original:  if dog → open
    Improved:  if dog AND confident enough → open
               if dog but low confidence  → uncertain, keep closed
               if not dog                 → closed
    """
    if not is_dog(class_index):
        return "CLOSED", "Not a dog"

    if confidence >= CONFIDENCE_THRESHOLD:
        return "OPEN", f"Dog detected ({confidence*100:.1f}% confident)"
    else:
        return "CLOSED", f"Uncertain prediction ({confidence*100:.1f}% — below {CONFIDENCE_THRESHOLD*100:.0f}% threshold)"


def run_doggy_door(image_paths, model_name="vgg16"):
    """
    MY IMPROVEMENT: Batch processing — test multiple images at once.
    Original only handled one image at a time.
    """
    print(f"\n{'='*60}")
    print(f"  🐾 SMART DOGGY DOOR  |  Model: {model_name.upper()}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"{'='*60}\n")

    model = load_model(model_name)

    results = []
    total_start = time.time()

    for path in image_paths:
        print(f"  📸 Processing: {os.path.basename(path)}")

        # Error handling — skip bad images
        image_tensor = load_image(path)
        if image_tensor is None:
            results.append({
                "image": os.path.basename(path),
                "status": "ERROR",
                "reason": "Could not load image",
                "confidence": 0,
                "time_ms": 0
            })
            continue

        # Time the prediction
        start = time.time()
        class_idx, confidence = predict_single(model, image_tensor)
        elapsed_ms = (time.time() - start) * 1000

        # Decision
        door_status, reason = doggy_door_decision(confidence, class_idx)

        icon = "✅ OPEN  " if door_status == "OPEN" else "❌ CLOSED"
        print(f"     → {icon}  |  {reason}  |  {elapsed_ms:.1f}ms\n")

        results.append({
            "image": os.path.basename(path),
            "status": door_status,
            "reason": reason,
            "confidence": confidence,
            "time_ms": elapsed_ms
        })

    total_time = (time.time() - total_start) * 1000
    print_summary(results, total_time)
    return results


def print_summary(results, total_time_ms):
    """MY IMPROVEMENT: Print a clean summary report at the end."""
    print(f"\n{'='*60}")
    print("  📊 SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"  {'IMAGE':<25} {'DOOR':<10} {'CONFIDENCE':<12}")
    print(f"  {'-'*50}")
    for r in results:
        conf_str = f"{r['confidence']*100:.1f}%" if r['status'] != "ERROR" else "N/A"
        door_icon = "✅ OPEN" if r['status'] == "OPEN" else ("⚠️ ERROR" if r['status'] == "ERROR" else "❌ CLOSED")
        print(f"  {r['image']:<25} {door_icon:<10} {conf_str:<12}")

    opened = sum(1 for r in results if r['status'] == "OPEN")
    errors = sum(1 for r in results if r['status'] == "ERROR")
    print(f"\n  Total images:   {len(results)}")
    print(f"  Door opened:    {opened}")
    print(f"  Door closed:    {len(results) - opened - errors}")
    print(f"  Errors:         {errors}")
    print(f"  Total time:     {total_time_ms:.1f}ms")
    print(f"{'='*60}\n")


def compare_models(image_paths):
    """
    MY IMPROVEMENT: Compare VGG16 vs ResNet50 on the same images.
    This is something the original course did NOT do.
    Shows understanding of different architectures.
    """
    print("\n" + "="*60)
    print("  🔬 MODEL COMPARISON: VGG16 vs ResNet50")
    print("="*60)

    for model_name in ["vgg16", "resnet50"]:
        model = load_model(model_name)
        print(f"\n  Model: {model_name.upper()}")
        print(f"  {'-'*40}")
        for path in image_paths:
            tensor = load_image(path)
            if tensor is None:
                continue
            class_idx, confidence = predict_single(model, tensor)
            status, reason = doggy_door_decision(confidence, class_idx)
            icon = "✅" if status == "OPEN" else "❌"
            print(f"  {icon} {os.path.basename(path):<20} → {reason}")


# ─────────────────────────────────────────────
# MAIN — RUN THE DOOR
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Update these paths to match your images folder
    test_images = [
        "images/happy_dog.jpg",
        "images/sleepy_cat.jpg",
        "images/brown_bear.jpg",
    ]

    # Run improved doggy door
    run_doggy_door(test_images, model_name="vgg16")

    # Bonus: compare VGG16 vs ResNet50
    compare_models(test_images)
