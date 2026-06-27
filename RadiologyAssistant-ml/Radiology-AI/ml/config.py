"""
config.py — Central configuration for all 5 steps
===================================================
Edit THIS file only when changing paths or settings.
All 5 steps import from here — so one change updates everything.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# HARD DRIVE PATH — set this to your external HDD drive letter
# Examples:
#   Windows HDD:  "E:/RadiologyData"
#   Windows HDD:  "F:/RadiologyData"
#   Same drive:   None   ← uses the ml/ folder inside your project
# ─────────────────────────────────────────────────────────────────────────────
HDD_PATH = "G:\RadiologyAssistant-ML"   # <-- CHANGE THIS to your HDD path e.g. "E:/RadiologyData"


# ─────────────────────────────────────────────────────────────────────────────
# Paths — automatically set based on HDD_PATH above
# ─────────────────────────────────────────────────────────────────────────────
ML_DIR = os.path.dirname(os.path.abspath(__file__))

if HDD_PATH:
    DATA_DIR      = os.path.join(HDD_PATH, "data")       # raw downloaded data
    PROCESSED_DIR = os.path.join(HDD_PATH, "processed")  # preprocessed .npy files
    MODEL_DIR     = os.path.join(HDD_PATH, "models")     # saved model weights
else:
    DATA_DIR      = os.path.join(ML_DIR, "data")
    PROCESSED_DIR = os.path.join(ML_DIR, "processed")
    MODEL_DIR     = os.path.join(ML_DIR, "models")

# Create directories if they don't exist
for d in [DATA_DIR, PROCESSED_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset settings
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 224    # pixels — standard input size for DenseNet121
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# The 5 chest conditions we detect
CLASSES = [
    "No Finding",
    "Pneumonia",
    "Pleural Effusion",
    "Cardiomegaly",
    "Atelectasis",
]


# ─────────────────────────────────────────────────────────────────────────────
# Training settings
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.00001


# ─────────────────────────────────────────────────────────────────────────────
# Print current config (called by each step on startup)
# ─────────────────────────────────────────────────────────────────────────────
def print_config():
    print("  Config:")
    print(f"    Data dir:      {DATA_DIR}")
    print(f"    Processed dir: {PROCESSED_DIR}")
    print(f"    Model dir:     {MODEL_DIR}")
    print(f"    HDD mode:      {'YES — ' + HDD_PATH if HDD_PATH else 'NO — using project folder'}")
    print()
