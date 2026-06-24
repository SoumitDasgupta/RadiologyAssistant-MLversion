"""
STEP 1 — DOWNLOAD TRAINING DATA
================================
This script downloads the CheXpert "small" dataset from Stanford.
It contains 224,316 chest X-rays with labels like:
  - Pneumonia
  - Pleural Effusion
  - Cardiomegaly
  - Atelectasis
  - No Finding (healthy)

You only need to run this ONCE.
It will create a folder called: ml/data/

HOW TO RUN:
  cd Radiology-AI/ml
  python step1_download_data.py
"""

import os
import zipfile
import urllib.request

# ── Where to save everything ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── What we are downloading ───────────────────────────────────────────────────
# CheXpert-v1.0-small is a lighter version (~11 GB) of the full dataset.
# It has two resolutions: 320x320 frontal/lateral views.
# Stanford provides it free for research use.
CHEXPERT_URL = "https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"
DOWNLOAD_PAGE = "https://stanfordmlgroup.github.io/competitions/chexpert/"

print("=" * 60)
print("  Radiology ML — Step 1: Training Data")
print("=" * 60)
print()
print("CheXpert requires a FREE registration before download.")
print("This is a legal requirement from Stanford University.")
print()
print("Please follow these steps:")
print()
print("  1. Open this URL in your browser:")
print(f"     {DOWNLOAD_PAGE}")
print()
print("  2. Fill in the form with your name and email.")
print()
print("  3. You will receive a download link by email.")
print()
print("  4. Download 'CheXpert-v1.0-small.zip' (about 11 GB)")
print()
print("  5. Place the zip file here:")
print(f"     {DATA_DIR}/CheXpert-v1.0-small.zip")
print()
print("  6. Then run this script again — it will extract it for you.")
print()

# ── If the zip already exists, extract it ────────────────────────────────────
zip_path = os.path.join(DATA_DIR, "CheXpert-v1.0-small.zip")

if os.path.exists(zip_path):
    print("Found zip file! Extracting now...")
    print("(This may take a few minutes)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Show progress every 1000 files
        all_files = zf.namelist()
        total = len(all_files)
        for i, name in enumerate(all_files):
            zf.extract(name, DATA_DIR)
            if i % 1000 == 0:
                pct = int((i / total) * 100)
                print(f"  Extracting... {pct}% ({i}/{total} files)", end="\r")

    print()
    print()
    print("✓ Extraction complete!")
    print(f"  Data is now in: {DATA_DIR}/CheXpert-v1.0-small/")
    print()
    print("Next step: run  python step2_preprocess.py")

else:
    print("─" * 60)
    print("Waiting for zip file at:")
    print(f"  {zip_path}")
    print()
    print("ALTERNATIVE — use a smaller demo dataset instead:")
    print("  If you just want to test the pipeline without downloading")
    print("  11 GB, run this to create 200 fake sample images:")
    print()
    print("    python step1_download_data.py --demo")
    print()

    # ── Demo mode: create fake data for testing ───────────────────────────────
    import sys
    if "--demo" in sys.argv:
        print("Creating demo dataset (200 synthetic images)...")

        try:
            from PIL import Image, ImageDraw
            import random, csv
        except ImportError:
            print("Installing Pillow...")
            os.system("pip install pillow --break-system-packages -q")
            from PIL import Image, ImageDraw
            import random, csv

        # The 5 disease classes we will train on
        CLASSES = [
            "No Finding",
            "Pneumonia",
            "Pleural Effusion",
            "Cardiomegaly",
            "Atelectasis",
        ]

        demo_dir = os.path.join(DATA_DIR, "demo")
        img_dir  = os.path.join(demo_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        rows = []  # will become our CSV label file

        for i in range(200):
            # Create a grayscale image that looks vaguely like an X-ray
            img = Image.new("L", (320, 320), color=random.randint(10, 40))
            draw = ImageDraw.Draw(img)

            # Draw a rough "ribcage" shape with white arcs
            for rib in range(5):
                y_start = 60 + rib * 40
                brightness = random.randint(180, 240)
                draw.arc(
                    [60, y_start, 260, y_start + 30],
                    start=0, end=180,
                    fill=brightness, width=3
                )

            # Draw a rough "heart" blob in the centre
            draw.ellipse([120, 100, 200, 180], fill=random.randint(100, 160))

            # Add some noise (random bright pixels = X-ray grain)
            for _ in range(500):
                x = random.randint(0, 319)
                y = random.randint(0, 319)
                img.putpixel((x, y), random.randint(180, 255))

            fname = f"patient_{i:04d}.jpg"
            img.save(os.path.join(img_dir, fname))

            # Randomly assign one label
            label = CLASSES[i % len(CLASSES)]
            rows.append({"filename": fname, "label": label})

        # Write CSV label file
        csv_path = os.path.join(demo_dir, "labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Created 200 demo images in: {img_dir}")
        print(f"✓ Created label file:         {csv_path}")
        print()
        print("Next step: run  python step2_preprocess.py --demo")
