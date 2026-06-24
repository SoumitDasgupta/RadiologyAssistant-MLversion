"""
STEP 4 — EVALUATE THE MODEL
=============================
After training, we need to know HOW GOOD the model actually is.
Accuracy alone is not enough for medical AI.

We calculate:
  - Accuracy        — what % of predictions were correct overall
  - Sensitivity     — of all REAL disease cases, how many did we CATCH?
                      (missing a disease = dangerous false negative)
  - Specificity     — of all HEALTHY cases, how many did we correctly clear?
                      (wrongly flagging healthy = unnecessary stress/tests)
  - AUC-ROC         — overall model quality, 0.5 = random, 1.0 = perfect
  - Confusion matrix — full table of what was predicted vs what was real

For medical AI, sensitivity is the most important metric.
  We would rather flag a healthy person for extra review
  than miss a real disease.

HOW TO RUN:
  python step4_evaluate.py          # real data
  python step4_evaluate.py --demo   # demo data

WHAT YOU GET:
  ml/models/evaluation_report.json  ← all metrics
  ml/models/confusion_matrix.txt    ← readable table
"""

import os
import sys
import json

# ── Install libraries if needed ───────────────────────────────────────────────
try:
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    import torchvision.models as models
    import torch.nn as nn
    print("✓ Libraries ready")
except ImportError:
    os.system("pip install torch torchvision numpy --break-system-packages -q")
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    import torchvision.models as models
    import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR        = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(ML_DIR, "processed")
MODEL_DIR     = os.path.join(ML_DIR, "models")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reuse the Dataset class from step 3 ──────────────────────────────────────
# (copy here so step4 can run independently)
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        arr    = np.load(sample["_path"])
        arr    = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).float()
        mean   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        label  = torch.tensor(sample["class_id"], dtype=torch.long)
        return tensor, label


# ── Metrics functions ─────────────────────────────────────────────────────────

def compute_confusion_matrix(all_true, all_pred, num_classes):
    """
    A confusion matrix shows every combination of (true class, predicted class).

    Example for 3 classes (0=Normal, 1=Pneumonia, 2=Effusion):

                  Predicted:
                  Normal  Pneumonia  Effusion
    Actual: Normal   [45]      [3]       [2]
            Pneumonia [ 5]     [38]      [7]
            Effusion  [ 2]      [4]      [44]

    The diagonal (top-left to bottom-right) = correct predictions.
    Off-diagonal = mistakes.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_true, all_pred):
        matrix[true][pred] += 1
    return matrix


def compute_per_class_metrics(matrix, classes):
    """
    For each class, compute:

    True Positive  (TP): model said YES, and it was YES
    True Negative  (TN): model said NO,  and it was NO
    False Positive (FP): model said YES, but it was NO  ← false alarm
    False Negative (FN): model said NO,  but it was YES ← dangerous miss

    Sensitivity (Recall) = TP / (TP + FN)
      → of all real cases, what fraction did we detect?

    Specificity = TN / (TN + FP)
      → of all healthy cases, what fraction did we correctly clear?

    Precision = TP / (TP + FP)
      → of all our positive predictions, what fraction were correct?

    F1 Score = harmonic mean of Precision and Sensitivity
      → overall balance between catching disease and avoiding false alarms
    """
    num_classes = len(classes)
    metrics     = {}

    for i, cls_name in enumerate(classes):
        TP = matrix[i][i]
        FP = matrix[:, i].sum() - TP     # everything predicted as i but wasn't
        FN = matrix[i, :].sum() - TP     # everything actually i but predicted as something else
        TN = matrix.sum() - TP - FP - FN

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1          = (2 * precision * sensitivity / (precision + sensitivity)
                       if (precision + sensitivity) > 0 else 0.0)

        metrics[cls_name] = {
            "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision":   round(precision, 4),
            "f1_score":    round(f1, 4),
        }

    return metrics


def compute_auc_roc(all_probs, all_true, num_classes):
    """
    AUC-ROC = Area Under the Receiver Operating Characteristic Curve.

    For each class:
      - ROC curve plots sensitivity vs (1 - specificity) at every threshold
      - AUC is the area under that curve
      - 0.5 = model is no better than random guessing
      - 1.0 = model is perfect

    We compute one AUC per class (one-vs-rest approach).
    """
    auc_scores = {}

    for class_idx in range(num_classes):
        # Binary: is this sample class_idx or not?
        binary_true  = [1 if t == class_idx else 0 for t in all_true]
        class_probs  = [p[class_idx] for p in all_probs]

        # Sort by probability descending
        pairs = sorted(zip(class_probs, binary_true), reverse=True)

        # Count positives and negatives
        n_pos = sum(binary_true)
        n_neg = len(binary_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            auc_scores[class_idx] = 0.5  # can't compute, default to random
            continue

        # Trapezoidal AUC calculation
        tp = 0
        fp = 0
        auc = 0.0
        prev_fp = 0
        prev_tp = 0

        for prob, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += (tp + prev_tp) / 2   # trapezoid height
                prev_tp = tp
                prev_fp = fp

        auc = auc / (n_pos * n_neg)  # normalize
        auc_scores[class_idx] = round(float(auc), 4)

    return auc_scores


def format_confusion_matrix(matrix, classes):
    """Format confusion matrix as a readable text table."""
    n   = len(classes)
    col = 12   # column width

    lines = ["Confusion Matrix", "=" * (col * (n + 1) + 4), ""]
    header = " " * col + "".join(f"{c[:col]:>{col}}" for c in classes)
    lines.append("Predicted →")
    lines.append(header)
    lines.append("-" * (col * (n + 1) + 4))

    for i, row_name in enumerate(classes):
        row_str = f"{row_name[:col-2]:<{col-2}}│ "
        for j in range(n):
            cell  = str(matrix[i][j])
            mark  = " ✓" if i == j else ""   # ✓ on diagonal = correct
            row_str += f"{(cell + mark):>{col}}"
        lines.append(row_str)

    lines.append("")
    lines.append("Rows = Actual class | Columns = Predicted class")
    lines.append("✓ marks correct predictions (diagonal)")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    demo_mode = "--demo" in sys.argv
    print("=" * 60)
    print("  Radiology ML — Step 4: Evaluate Model")
    print("=" * 60)
    print()

    # ── 1. Load manifest ──────────────────────────────────────────────────────
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No preprocessed data. Run step2 first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    CLASSES     = manifest["classes"]
    NUM_CLASSES = len(CLASSES)

    # Attach full paths
    test_dir  = os.path.join(PROCESSED_DIR, "test")
    test_data = manifest["data"]["test"]
    for sample in test_data:
        sample["_path"] = os.path.join(test_dir, sample["file"])

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "radiology_model.pth")
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run step3 first.")
        return

    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, NUM_CLASSES)
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    best_val_acc = checkpoint.get("val_acc", "N/A")
    print(f"  Loaded model (best val accuracy: {best_val_acc}%)")
    print()

    # ── 3. Run inference on test set ──────────────────────────────────────────
    print("Running inference on test set...")
    test_dataset = XRayDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_true  = []
    all_pred  = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)

            # Convert raw scores to probabilities using softmax
            # softmax ensures all 5 class scores sum to 1.0
            probs     = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted = outputs.argmax(dim=1).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(predicted.tolist())
            all_probs.extend(probs.tolist())

    total    = len(all_true)
    correct  = sum(1 for t, p in zip(all_true, all_pred) if t == p)
    accuracy = correct / total * 100

    print(f"  Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print()

    # ── 4. Compute all metrics ────────────────────────────────────────────────
    print("Computing metrics...")

    # Confusion matrix
    matrix       = compute_confusion_matrix(all_true, all_pred, NUM_CLASSES)

    # Per-class sensitivity, specificity, precision, F1
    class_metrics = compute_per_class_metrics(matrix, CLASSES)

    # AUC-ROC per class
    auc_scores    = compute_auc_roc(all_probs, all_true, NUM_CLASSES)

    # ── 5. Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()

    print(f"  Overall Accuracy: {accuracy:.1f}%")
    print()

    header = f"  {'Class':<22} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8} {'AUC':>8}"
    print(header)
    print("  " + "-" * 62)

    for i, cls in enumerate(CLASSES):
        m   = class_metrics[cls]
        auc = auc_scores.get(i, 0.0)
        bar = "█" * int(m["sensitivity"] * 20)
        flag = ""
        if m["sensitivity"] < 0.7:
            flag = " ⚠ LOW SENSITIVITY"   # this is the dangerous one
        print(f"  {cls:<22} {m['sensitivity']:>12.3f} {m['specificity']:>12.3f} "
              f"{m['f1_score']:>8.3f} {auc:>8.3f}{flag}")

    print()
    print("  What these numbers mean:")
    print("  Sensitivity = fraction of REAL cases we correctly detected")
    print("  Specificity = fraction of HEALTHY cases we correctly cleared")
    print("  AUC 1.0 = perfect, 0.5 = no better than guessing")
    print()

    # Print confusion matrix
    cm_text = format_confusion_matrix(matrix, CLASSES)
    print(cm_text)

    # ── 6. Save everything ────────────────────────────────────────────────────
    report = {
        "overall_accuracy": round(accuracy, 2),
        "per_class":        {
            cls: {**class_metrics[cls], "auc_roc": auc_scores.get(i, 0.0)}
            for i, cls in enumerate(CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
        "classes":          CLASSES,
        "total_test_samples": total,
        "interpretation": {
            "sensitivity": "Fraction of real disease cases correctly detected. Most critical for safety.",
            "specificity": "Fraction of healthy cases correctly cleared. High = fewer false alarms.",
            "auc_roc":     "Overall model quality. 1.0 = perfect, 0.5 = random.",
            "warning":     "AI predictions must always be confirmed by a qualified radiologist."
        }
    }

    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.txt")
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(cm_text)

    print()
    print(f"✓ Evaluation report saved: {report_path}")
    print(f"✓ Confusion matrix saved:  {cm_path}")
    print()
    print("Next step: run  python step5_replace_gemini.py --test")


if __name__ == "__main__":
    main()
