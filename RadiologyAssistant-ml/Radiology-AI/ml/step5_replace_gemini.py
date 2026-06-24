"""
STEP 5 — REPLACE GEMINI WITH YOUR TRAINED MODEL
=================================================
This is the final step. We replace the Gemini API call in ai.py
with our trained DenseNet121 model.

BEFORE (current ai.py):
  - Sends image to Google's Gemini API
  - Gemini generates text → we parse JSON from it
  - Hallucination risk: high (it's a language model guessing)
  - Cost: per API call
  - Privacy: image leaves your server

AFTER (this file):
  - Image runs through YOUR trained model locally
  - Model outputs 5 probabilities directly
  - Hallucination risk: ZERO (it's pure maths, no language generation)
  - Cost: free (runs on your own hardware)
  - Privacy: image never leaves your server

HOW TO USE:
  1. Make sure step3 (training) has been completed.
     The file ml/models/radiology_model.pth must exist.

  2. Test the new analyzer:
       python step5_replace_gemini.py --test path/to/xray.jpg

  3. To switch your app to use this instead of Gemini:
       In backend/ai.py, replace the import at the top:
         from ai import analyze_image
       with:
         from ml.step5_replace_gemini import analyze_image

  Or copy the analyze_image function below directly into backend/ai.py.
"""

import os
import sys
import json

# ── Install libraries if needed ───────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    import numpy as np
    from PIL import Image
    print("✓ Libraries ready")
except ImportError:
    os.system("pip install torch torchvision pillow numpy --break-system-packages -q")
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    import numpy as np
    from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, "models", "radiology_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 5 conditions our model was trained on ────────────────────────────────────
CLASSES = [
    "No Finding",        # 0 — healthy / normal
    "Pneumonia",         # 1
    "Pleural Effusion",  # 2
    "Cardiomegaly",      # 3
    "Atelectasis",       # 4
]

# Clinical descriptions for each finding (shown to the user)
CLASS_DETAILS = {
    "No Finding": {
        "details":        "No significant acute cardiopulmonary findings identified.",
        "recommendation": "Routine follow-up as clinically indicated. No urgent action required.",
    },
    "Pneumonia": {
        "details":        "Increased opacity or consolidation pattern in the lung fields, "
                          "suggesting possible infectious or inflammatory process.",
        "recommendation": "Clinical correlation recommended. Consider antibiotic therapy if "
                          "clinically suspected. Follow-up imaging in 4-6 weeks to confirm resolution.",
    },
    "Pleural Effusion": {
        "details":        "Blunting of the costophrenic angle or increased opacity at the lung base, "
                          "suggesting fluid accumulation in the pleural space.",
        "recommendation": "Evaluate for underlying cause (cardiac, hepatic, malignant). "
                          "Thoracentesis may be considered if clinically indicated.",
    },
    "Cardiomegaly": {
        "details":        "Cardiothoracic ratio appears increased, suggesting possible cardiac enlargement.",
        "recommendation": "Echocardiography recommended for further cardiac evaluation. "
                          "Correlate with clinical symptoms and ECG findings.",
    },
    "Atelectasis": {
        "details":        "Linear or plate-like densities visible, possibly representing subsegmental "
                          "or segmental atelectasis.",
        "recommendation": "Incentive spirometry and deep breathing exercises recommended. "
                          "Follow-up imaging if symptoms persist.",
    },
}

# ── Model loading (done once, reused for every request) ───────────────────────
# We load the model once at module import time — not on every request.
# This means the first request may take a second, but all subsequent ones are fast.

_model      = None    # will hold the loaded model
_model_info = {}      # metadata from the checkpoint


def _load_model():
    """
    Load the trained model from disk.
    Called automatically on first use.
    """
    global _model, _model_info

    if _model is not None:
        return _model    # already loaded

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Please run step3_train.py first."
        )

    print(f"Loading trained model from {MODEL_PATH}...")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Rebuild the same architecture we used in step3
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, len(CLASSES))
    )

    # Load the learned weights
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()    # evaluation mode: disables dropout

    _model_info = {
        "epoch":   checkpoint.get("epoch", "unknown"),
        "val_acc": checkpoint.get("val_acc", "unknown"),
    }

    _model = model
    print(f"✓ Model loaded (trained for {_model_info['epoch']} epochs, "
          f"val accuracy: {_model_info['val_acc']}%)")
    return model


def preprocess_image_for_inference(image_path):
    """
    Prepare a raw X-ray image for model input.

    This mirrors the preprocessing in step2, but without saving to disk.
    We apply the same steps live on each incoming image.
    """
    # 1. Open as grayscale
    img = Image.open(image_path).convert("L")

    # 2. Resize to 224×224
    img = img.resize((224, 224), Image.LANCZOS)

    # 3. Apply simplified CLAHE (contrast enhancement)
    img_array = np.array(img, dtype=np.float32)
    p2  = np.percentile(img_array, 2)
    p98 = np.percentile(img_array, 98)
    if p98 > p2:
        img_array = np.clip(img_array, p2, p98)
        img_array = (img_array - p2) / (p98 - p2)
    else:
        img_array = img_array / 255.0

    # 4. Stack to 3 channels (model expects RGB)
    img_array = np.stack([img_array, img_array, img_array], axis=0)  # (3, 224, 224)

    # 5. Normalize with ImageNet statistics (same as during training)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std

    # 6. Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    tensor = torch.from_numpy(img_array).float().unsqueeze(0)

    return tensor


def analyze_image(image_path):
    """
    Main function — replaces the Gemini analyze_image() in backend/ai.py.

    Takes:  image_path (string) — path to the X-ray image file
    Returns: JSON string with the same structure as the Gemini version

    The output format is IDENTICAL to what Gemini was returning,
    so your backend/app.py does NOT need any changes.
    """
    try:
        # 1. Load model (cached after first call)
        model = _load_model()

        # 2. Preprocess image
        tensor = preprocess_image_for_inference(image_path)
        tensor = tensor.to(DEVICE)

        # 3. Run model inference
        with torch.no_grad():
            outputs = model(tensor)                           # raw scores (logits)
            probs   = torch.softmax(outputs, dim=1)           # convert to probabilities
            probs   = probs.cpu().numpy()[0]                  # numpy array of 5 values

        # 4. Get top prediction
        predicted_class_id    = int(np.argmax(probs))
        predicted_class       = CLASSES[predicted_class_id]
        predicted_probability = float(probs[predicted_class_id])

        # 5. Convert probability to confidence score (0-100 integer)
        #    We apply a slight calibration: very high raw probabilities
        #    are capped at 92 to avoid overconfidence in medical context
        confidence = min(int(predicted_probability * 100), 92)

        # 6. Build all_findings: list of all classes with their probabilities
        #    (shown to the radiologist so they can see the full picture)
        all_findings = []
        for i, cls in enumerate(CLASSES):
            if i == predicted_class_id:
                continue    # primary finding is shown separately
            pct = round(float(probs[i]) * 100, 1)
            if pct > 2.0:   # only show if probability > 2%
                all_findings.append({
                    "title":   cls,
                    "details": f"Probability: {pct}% — {CLASS_DETAILS[cls]['details']}"
                })

        # 7. Build the JSON response (same structure as Gemini was returning)
        info     = CLASS_DETAILS.get(predicted_class, CLASS_DETAILS["No Finding"])
        response = {
            "findings": (
                f"ML Model Analysis — Primary finding: {predicted_class} "
                f"(confidence: {confidence}%). "
                f"{info['details']}"
            ),
            "findings_list": [
                {
                    "title":   predicted_class,
                    "details": (
                        f"Confidence: {confidence}% | "
                        f"Raw probability: {predicted_probability:.3f} | "
                        f"{info['details']}"
                    )
                }
            ] + all_findings[:4],   # include up to 4 secondary findings
            "impression": (
                f"The ML model classifies this image as: {predicted_class} "
                f"with {confidence}% confidence. "
                f"This is an automated analysis — radiologist review is mandatory."
            ),
            "confidence": confidence,
            "recommendation": info["recommendation"],
            "recommendations_list": [
                info["recommendation"],
                "This analysis was generated by a machine learning model, "
                "not a qualified radiologist.",
                "Clinical correlation and radiologist review are mandatory "
                "before any clinical decision.",
                f"Model training: DenseNet121 fine-tuned on chest X-ray dataset. "
                f"Trained accuracy: {_model_info.get('val_acc', 'N/A')}%"
            ],
            # Extra field: full probability breakdown (not in Gemini version)
            # Your frontend can use this to show a probability bar chart
            "class_probabilities": {
                cls: round(float(probs[i]) * 100, 2)
                for i, cls in enumerate(CLASSES)
            }
        }

        return json.dumps(response)

    except FileNotFoundError as e:
        # Model not trained yet — fall back gracefully
        return json.dumps({
            "error": str(e),
            "findings": "ML model not yet trained.",
            "findings_list": [],
            "impression": "Model file not found. Please complete training first.",
            "confidence": 0,
            "recommendation": "Run step3_train.py to train the model.",
            "recommendations_list": ["Train the model before use."]
        })
    except Exception as e:
        return json.dumps({
            "error":       f"Inference error: {str(e)}",
            "findings":    "Analysis failed.",
            "findings_list": [],
            "impression":  "An error occurred during ML inference.",
            "confidence":  0,
            "recommendation": "Check image format and model integrity.",
            "recommendations_list": [f"Error details: {str(e)}"]
        })


# ── Test mode (run this file directly) ────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Radiology ML — Step 5: ML Inference Test")
    print("=" * 60)
    print()

    # Look for a test image
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        test_image = sys.argv[2]
    else:
        # Try to find one of the sample images from the project
        uploads_dir = os.path.join(ML_DIR, "..", "uploads")
        test_images = []
        if os.path.exists(uploads_dir):
            for f in os.listdir(uploads_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
                    test_images.append(os.path.join(uploads_dir, f))

        if test_images:
            test_image = test_images[0]
            print(f"Using sample image: {os.path.basename(test_image)}")
        else:
            print("Usage: python step5_replace_gemini.py --test path/to/xray.jpg")
            print()
            print("No test image found. To use this module in your app:")
            print("  Replace 'from ai import analyze_image' in backend/app.py")
            print("  with:   from ml.step5_replace_gemini import analyze_image")
            sys.exit(0)

    print(f"Analyzing: {test_image}")
    print()

    result = analyze_image(test_image)
    parsed = json.loads(result)

    if "error" in parsed and not "class_probabilities" in parsed:
        print(f"Error: {parsed['error']}")
    else:
        print(f"Primary finding:  {parsed['findings_list'][0]['title'] if parsed['findings_list'] else 'N/A'}")
        print(f"Confidence:       {parsed['confidence']}%")
        print(f"Impression:       {parsed['impression']}")
        print()
        if "class_probabilities" in parsed:
            print("Class probabilities:")
            for cls, pct in sorted(parsed["class_probabilities"].items(),
                                   key=lambda x: x[1], reverse=True):
                bar = "█" * int(pct / 5)
                print(f"  {cls:<22} {pct:>6.2f}%  {bar}")
        print()
        print("✓ ML inference working correctly.")
        print()
        print("To activate in your app:")
        print("  Open backend/ai.py and replace everything with:")
        print("    from ml.step5_replace_gemini import analyze_image")
