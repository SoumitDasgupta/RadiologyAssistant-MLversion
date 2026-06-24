"""
STEP 2 — PREPROCESS & AUGMENT IMAGES
======================================
Raw X-ray images can't go straight into a model. We need to:

  1. RESIZE       — all images must be the same size (224×224 pixels)
  2. NORMALIZE    — pixel values scaled from 0-255 to 0.0-1.0
  3. CLAHE        — contrast enhancement specific to medical images
  4. AUGMENT      — flip/rotate images to artificially grow the dataset
  5. SPLIT        — divide into train (70%), validation (15%), test (15%)

Think of this like preparing ingredients before cooking.
Raw = unprepared. After this step = ready for the model.

HOW TO RUN:
  python step2_preprocess.py          # real CheXpert data
  python step2_preprocess.py --demo   # demo data from step 1
"""

import os
import csv
import sys
import json
import shutil
import random
from pathlib import Path

# We will use these libraries — install if missing
try:
    import numpy as np
    from PIL import Image, ImageOps, ImageFilter
    print("✓ Libraries already installed")
except ImportError:
    print("Installing required libraries...")
    os.system("pip install numpy pillow --break-system-packages -q")
    import numpy as np
    from PIL import Image, ImageOps, ImageFilter

# ── Configuration ─────────────────────────────────────────────────────────────
ML_DIR      = os.path.dirname(__file__)
DATA_DIR    = os.path.join(ML_DIR, "data")
OUTPUT_DIR  = os.path.join(ML_DIR, "processed")

IMAGE_SIZE  = 224          # pixels × pixels (standard for most vision models)
TRAIN_SPLIT = 0.70         # 70% of data for training
VAL_SPLIT   = 0.15         # 15% for validation (tuning)
TEST_SPLIT  = 0.15         # 15% for final testing (never seen during training)

# The 5 conditions we classify — these become class IDs 0-4
CLASSES = [
    "No Finding",          # healthy — class 0
    "Pneumonia",           # class 1
    "Pleural Effusion",    # class 2
    "Cardiomegaly",        # class 3
    "Atelectasis",         # class 4
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_clahe(img_array):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Normal X-rays can be too dark or too bright in patches.
    CLAHE enhances local contrast so the model can see detail
    in both dark lung fields AND bright bone areas.

    Think of it like auto-levels in Photoshop, but smarter.
    """
    # Split the image into an 8×8 grid of tiles
    h, w = img_array.shape
    tile_h = h // 8
    tile_w = w // 8
    result = img_array.copy().astype(np.float32)

    for row in range(8):
        for col in range(8):
            # Extract this tile
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            tile = result[y0:y1, x0:x1]

            # Equalize histogram of this tile
            tile_min = tile.min()
            tile_max = tile.max()
            if tile_max > tile_min:
                # Clip extreme values (the "limited" part of CLAHE)
                p2  = np.percentile(tile, 2)
                p98 = np.percentile(tile, 98)
                tile = np.clip(tile, p2, p98)
                # Rescale to 0-255
                tile = (tile - p2) / (p98 - p2) * 255.0
            result[y0:y1, x0:x1] = tile

    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_image(img_path):
    """
    Takes a raw X-ray image file path.
    Returns a clean numpy array ready for the model.

    Steps:
      1. Open as grayscale (X-rays are black & white)
      2. Resize to 224×224
      3. Apply CLAHE contrast enhancement
      4. Normalize to 0.0-1.0 range
      5. Convert to 3-channel (models expect RGB even for grayscale)
    """
    # 1. Open image
    img = Image.open(img_path).convert("L")   # "L" = grayscale

    # 2. Resize — LANCZOS gives best quality for downscaling
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # 3. CLAHE contrast enhancement
    img_array = np.array(img)
    img_array = apply_clahe(img_array)

    # 4. Normalize: pixels go from 0-255 → 0.0-1.0
    img_array = img_array.astype(np.float32) / 255.0

    # 5. Stack 3 copies to make "RGB" (model expects 3 channels)
    #    Shape goes from (224, 224) → (224, 224, 3)
    img_array = np.stack([img_array, img_array, img_array], axis=-1)

    return img_array


def augment_image(img_array):
    """
    Data augmentation — creates variations of an image so the model
    learns to recognize findings regardless of orientation/brightness.

    Like showing a student the same chest X-ray from slightly different
    angles so they learn the pattern, not just one specific view.

    Returns a list of augmented versions (including the original).
    """
    versions = [img_array]   # always include original

    # Convert to PIL for easy transforms
    # img_array is (224, 224, 3) float32 → need uint8 PIL
    pil = Image.fromarray((img_array[:, :, 0] * 255).astype(np.uint8), mode="L")

    # Flip horizontally (left-right mirror)
    flipped = ImageOps.mirror(pil)
    flipped_arr = np.array(flipped).astype(np.float32) / 255.0
    versions.append(np.stack([flipped_arr]*3, axis=-1))

    # Slight rotation (±5 degrees — more than this distorts anatomy)
    for angle in [-5, 5]:
        rotated = pil.rotate(angle, fillcolor=0)
        rot_arr = np.array(rotated).astype(np.float32) / 255.0
        versions.append(np.stack([rot_arr]*3, axis=-1))

    # Slight brightness change (simulate different X-ray exposures)
    for brightness in [0.9, 1.1]:
        bright = np.clip(img_array * brightness, 0, 1)
        versions.append(bright)

    return versions  # 5 versions total per image


def load_chexpert_labels(data_dir):
    """
    CheXpert labels come in a CSV file with columns like:
      Path, No Finding, Enlarged Cardiomediastinum, Cardiomegaly, ...
    
    Values: 1.0 = present, 0.0 = absent, -1.0 = uncertain
    We treat uncertain as absent (conservative approach for safety).
    """
    csv_path = os.path.join(data_dir, "CheXpert-v1.0-small", "train.csv")
    samples = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(data_dir, row["Path"])
            if not os.path.exists(img_path):
                continue

            # Map to our 5 classes
            label = "No Finding"
            if float(row.get("Pneumonia", 0) or 0) == 1.0:
                label = "Pneumonia"
            elif float(row.get("Pleural Effusion", 0) or 0) == 1.0:
                label = "Pleural Effusion"
            elif float(row.get("Cardiomegaly", 0) or 0) == 1.0:
                label = "Cardiomegaly"
            elif float(row.get("Atelectasis", 0) or 0) == 1.0:
                label = "Atelectasis"

            samples.append((img_path, label))

    return samples


def load_demo_labels(data_dir):
    """Load the demo CSV created by step1 --demo"""
    csv_path = os.path.join(data_dir, "demo", "labels.csv")
    img_dir  = os.path.join(data_dir, "demo", "images")
    samples  = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(img_dir, row["filename"])
            if os.path.exists(img_path):
                samples.append((img_path, row["label"]))

    return samples


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    demo_mode = "--demo" in sys.argv
    print("=" * 60)
    print("  Radiology ML — Step 2: Preprocess Images")
    print("=" * 60)
    print(f"  Mode: {'DEMO (200 images)' if demo_mode else 'FULL CheXpert dataset'}")
    print()

    # 1. Load labels
    print("Loading image paths and labels...")
    if demo_mode:
        samples = load_demo_labels(DATA_DIR)
    else:
        samples = load_chexpert_labels(DATA_DIR)

    if not samples:
        print("ERROR: No images found.")
        print("Run step1 first:  python step1_download_data.py --demo")
        return

    print(f"  Found {len(samples)} images")

    # Show class distribution
    from collections import Counter
    counts = Counter(label for _, label in samples)
    print("  Class distribution:")
    for cls, count in counts.most_common():
        bar = "█" * (count * 30 // max(counts.values()))
        print(f"    {cls:<22} {count:>5}  {bar}")
    print()

    # 2. Shuffle and split
    print("Splitting into train/val/test...")
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train : n_train + n_val],
        "test":  samples[n_train + n_val:],
    }
    for split, data in splits.items():
        print(f"  {split:<8} {len(data)} images")
    print()

    # 3. Process and save each split
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {}   # we'll save this as JSON for step 3 to read

    for split_name, split_samples in splits.items():
        print(f"Processing {split_name} set...")
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        manifest[split_name] = []
        count = 0

        for img_path, label in split_samples:
            try:
                # Preprocess
                arr = preprocess_image(img_path)

                # Augment only the training set (val/test stay clean)
                if split_name == "train":
                    versions = augment_image(arr)
                else:
                    versions = [arr]

                # Save each version as a .npy file (fast numpy format)
                for v_idx, version in enumerate(versions):
                    out_name = f"{count:06d}_v{v_idx}.npy"
                    out_path = os.path.join(split_dir, out_name)
                    np.save(out_path, version)

                    manifest[split_name].append({
                        "file":  out_name,
                        "label": label,
                        "class_id": CLASSES.index(label),
                    })

                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{len(split_samples)} processed...", end="\r")

            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
                continue

        print(f"  ✓ {split_name}: saved {len(manifest[split_name])} items")

    # 4. Save manifest JSON
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "splits": {k: len(v) for k, v in manifest.items()},
            "data": manifest,
        }, f, indent=2)

    print()
    print(f"✓ Manifest saved: {manifest_path}")
    print()
    print("Next step: run  python step3_train.py")


if __name__ == "__main__":
    main()
