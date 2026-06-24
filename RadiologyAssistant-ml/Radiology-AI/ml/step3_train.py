"""
STEP 3 — TRAIN THE MODEL
=========================
This is where the actual machine learning happens.

What we are doing:
  - Take DenseNet121 — a model already trained on millions of natural images
    (it already knows edges, shapes, textures)
  - Replace its last layer with our 5-class radiology classifier
  - Train it on our preprocessed X-ray images
  - Save the trained model to a file

This is called TRANSFER LEARNING.
  Analogy: You don't teach a radiologist to see from scratch.
           They already know what lines and shapes look like (from life).
           You just teach them what specific patterns mean medically.
           That's exactly what we're doing here.

HOW TO RUN:
  python step3_train.py          # real CheXpert data
  python step3_train.py --demo   # demo data from step 2

WHAT YOU GET:
  ml/models/radiology_model.pth   ← trained model weights (the "brain")
  ml/models/training_log.json     ← accuracy/loss per epoch
"""

import os
import sys
import json
import time
import random

# ── Install libraries if missing ──────────────────────────────────────────────
# PyTorch  = the ML framework (does the actual math)
# torchvision = vision-specific tools (DenseNet, image transforms)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as T
    import numpy as np
    print("✓ PyTorch already installed")
except ImportError:
    print("Installing PyTorch and torchvision (this may take a few minutes)...")
    os.system("pip install torch torchvision numpy --break-system-packages -q")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as T
    import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
ML_DIR        = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(ML_DIR, "processed")
MODEL_DIR     = os.path.join(ML_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters — these control HOW the model learns
BATCH_SIZE    = 16      # how many images to look at at once before updating
EPOCHS        = 3       # how many times to go through the full training set
LEARNING_RATE = 0.0001  # how big each learning step is (small = careful)
NUM_CLASSES   = 5       # No Finding, Pneumonia, Pleural Effusion, Cardiomegaly, Atelectasis

# Use GPU if available (much faster), otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("  (No GPU detected — training will be slower but still works)")


# ── Dataset class ─────────────────────────────────────────────────────────────

class XRayDataset(Dataset):
    """
    A Dataset tells PyTorch how to read one item of our data.

    PyTorch will call __getitem__(index) thousands of times during training.
    We return: (image_tensor, class_id) for each call.
    """

    def __init__(self, samples, transform=None):
        """
        samples  = list of {"file": "...", "label": "...", "class_id": 2}
        transform = optional image transformations (augmentation)
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        # PyTorch asks "how many items do you have?"
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load the preprocessed numpy array we saved in step 2
        npy_dir = os.path.join(PROCESSED_DIR, "train" if "train" in str(idx) else "")
        # We stored the split path in the manifest, so we look relative
        # The files are stored as (H, W, C) float32 arrays
        arr = np.load(sample["_path"])  # shape: (224, 224, 3)

        # PyTorch expects (C, H, W) not (H, W, C) — so we rearrange
        arr = arr.transpose(2, 0, 1)   # → (3, 224, 224)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(arr).float()

        # Apply any transforms (e.g. random augmentation)
        if self.transform:
            tensor = self.transform(tensor)

        # Normalize using ImageNet statistics
        # Why ImageNet? Because DenseNet was originally trained on ImageNet.
        # Using the same normalization keeps the pretrained weights working.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        label = torch.tensor(sample["class_id"], dtype=torch.long)

        return tensor, label


# ── Model definition ──────────────────────────────────────────────────────────

def build_model(num_classes):
    """
    Build our classifier on top of DenseNet121.

    DenseNet121 architecture:
      - 121 layers deep
      - Each layer connects to ALL previous layers (that's what Dense means)
      - Originally trained to classify 1000 ImageNet categories
      - The CheXNet paper (Stanford, 2017) showed it matches radiologists
        when fine-tuned on chest X-rays

    What we change:
      - Keep all 121 layers (they already detect useful patterns)
      - Replace ONLY the final classification layer
        Original: outputs 1000 class scores (ImageNet)
        Ours:     outputs 5 class scores (our conditions)
    """
    # Load DenseNet121 with pretrained ImageNet weights
    # pretrained=True downloads ~30MB of learned weights
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # How many features does DenseNet output before its final layer?
    num_features = model.classifier.in_features
    # Answer: 1024 features

    # Replace the final layer with our 5-class classifier
    # Linear(1024 → 5) means: take 1024 numbers, output 5 scores
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),            # randomly zero out 50% of features → reduces overfitting
        nn.Linear(num_features, 5) # map 1024 features → 5 class scores
    )

    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One epoch = one full pass through all training images.

    For each batch:
      1. Forward pass  — run images through the model, get predictions
      2. Compute loss  — how wrong were we? (cross-entropy)
      3. Backward pass — calculate which weights caused the error
      4. Update weights — move weights slightly in the right direction

    This is like a student:
      1. Reads a question
      2. Gives an answer
      3. Gets told what the right answer was
      4. Updates their understanding
    """
    model.train()           # tell model it's in training mode (activates dropout)
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)  # move data to GPU if available
        labels = labels.to(device)

        # 1. Forward pass
        optimizer.zero_grad()       # reset gradients from previous batch
        outputs = model(images)     # outputs shape: (batch_size, 5)

        # 2. Compute loss
        # CrossEntropyLoss: standard loss for multi-class classification
        # It measures how different our probability distribution is from truth
        loss = criterion(outputs, labels)

        # 3. Backward pass — PyTorch automatically calculates gradients
        loss.backward()

        # 4. Update weights (Adam optimizer takes a step in right direction)
        optimizer.step()

        # Track statistics
        total_loss    += loss.item()
        predicted      = outputs.argmax(dim=1)   # take the class with highest score
        correct       += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Show progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            running_acc = correct / total_samples * 100
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Accuracy: {running_acc:.1f}%", end="\r")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Run the model on validation/test data WITHOUT updating weights.
    This tells us how well the model generalises to new images it hasn't seen.
    """
    model.eval()            # tell model it's in eval mode (disables dropout)
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():   # don't track gradients (saves memory, speeds up)
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs    = model(images)
            loss       = criterion(outputs, labels)
            predicted  = outputs.argmax(dim=1)

            total_loss    += loss.item()
            correct       += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    demo_mode = "--demo" in sys.argv
    print("=" * 60)
    print("  Radiology ML — Step 3: Train the Model")
    print("=" * 60)
    print(f"  Mode:   {'DEMO' if demo_mode else 'FULL CheXpert'}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch:  {BATCH_SIZE}")
    print(f"  LR:     {LEARNING_RATE}")
    print()

    # ── 1. Load the manifest from step 2 ──────────────────────────────────────
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No preprocessed data found.")
        print("Run step 2 first:  python step2_preprocess.py --demo")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    CLASSES = manifest["classes"]
    print(f"  Classes: {CLASSES}")
    print()

    # ── 2. Attach full paths to each sample ───────────────────────────────────
    # The manifest stores filenames; we add the full disk path here
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        for sample in manifest["data"][split_name]:
            sample["_path"] = os.path.join(split_dir, sample["file"])

    # ── 3. Create Dataset objects ─────────────────────────────────────────────
    train_dataset = XRayDataset(manifest["data"]["train"])
    val_dataset   = XRayDataset(manifest["data"]["val"])
    test_dataset  = XRayDataset(manifest["data"]["test"])

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print()

    # ── 4. Create DataLoaders ─────────────────────────────────────────────────
    # DataLoader wraps a dataset and feeds it to the model in batches
    # shuffle=True randomises order each epoch (prevents the model memorising order)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 5. Build model ────────────────────────────────────────────────────────
    print("Building DenseNet121 model...")
    model = build_model(NUM_CLASSES)
    model = model.to(DEVICE)

    # Count trainable parameters — just for information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    print()

    # ── 6. Loss function and optimizer ────────────────────────────────────────
    # CrossEntropyLoss: standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: an improved version of gradient descent
    # It adapts the learning rate for each parameter automatically
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler: reduce LR by 50% if val loss stops improving
    # This helps the model fine-tune more carefully in later epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print("Starting training...")
    print("-" * 60)

    best_val_acc  = 0.0
    training_log  = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        print(f"\n  Train  — Loss: {train_loss:.4f} | Accuracy: {train_acc:.1f}%")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Val    — Loss: {val_loss:.4f} | Accuracy: {val_acc:.1f}%")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save this epoch's stats
        epoch_time = time.time() - epoch_start
        log_entry  = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 2),
            "time_sec":   round(epoch_time, 1),
        }
        training_log.append(log_entry)
        print(f"  Time: {epoch_time:.1f}s")

        # Save the best model (lowest validation loss = most generalised)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path   = os.path.join(MODEL_DIR, "radiology_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "classes":     CLASSES,
            }, model_path)
            print(f"  ✓ New best model saved! (val_acc: {val_acc:.1f}%)")

    # ── 8. Final test evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final evaluation on test set (data model has NEVER seen)...")

    # Load best model
    checkpoint = torch.load(os.path.join(MODEL_DIR, "radiology_model.pth"),
                            map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1f}%")

    # ── 9. Save training log ──────────────────────────────────────────────────
    log_path = os.path.join(MODEL_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "epochs":        EPOCHS,
                "batch_size":    BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "device":        str(DEVICE),
                "classes":       CLASSES,
            },
            "best_val_acc": round(best_val_acc, 2),
            "test_acc":     round(test_acc, 2),
            "epochs":       training_log,
        }, f, indent=2)

    print(f"\n✓ Training log saved: {log_path}")
    print(f"✓ Model saved:        {os.path.join(MODEL_DIR, 'radiology_model.pth')}")
    print()
    print("Next step: run  python step4_evaluate.py")


if __name__ == "__main__":
    main()
