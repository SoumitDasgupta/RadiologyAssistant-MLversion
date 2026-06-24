# Radiology AI — ML Training Guide
## From Gemini API → Your Own Trained Model

---

## What this does

Right now your app sends X-rays to Google's Gemini and asks it to
describe what it sees. Gemini is a language model — it can hallucinate.

After following these steps, your app will run its own trained
DenseNet121 model that was fine-tuned specifically on chest X-rays.
No API calls. No hallucination. Runs on your machine.

---

## The 5 steps (run in order)

### Step 1 — Get training data
```
cd Radiology-AI/ml
python step1_download_data.py --demo
```
This creates 200 synthetic X-ray-like images for testing the pipeline.
For real training, register at https://stanfordmlgroup.github.io/competitions/chexpert/
and download CheXpert (11 GB, free for research).

---

### Step 2 — Preprocess the images
```
python step2_preprocess.py --demo
```
Resizes all images to 224×224, applies CLAHE contrast enhancement,
and splits data into 70% train / 15% val / 15% test.

---

### Step 3 — Train the model
```
python step3_train.py --demo
```
Fine-tunes DenseNet121 on your data. Saves the best model to:
  `ml/models/radiology_model.pth`

With demo data (200 images): takes ~2 minutes on CPU.
With full CheXpert (224k images): takes ~4-8 hours on GPU.

---

### Step 4 — Evaluate
```
python step4_evaluate.py --demo
```
Calculates sensitivity, specificity, AUC-ROC, and confusion matrix.
Saves a report to `ml/models/evaluation_report.json`.

---

### Step 5 — Replace Gemini
```
python step5_replace_gemini.py --test ../uploads/test.jpg
```
Tests the ML analyzer on a real image. If it works, open
`backend/ai.py` and replace everything with one line:

```python
from ml.step5_replace_gemini import analyze_image
```

That's it. Your app now uses your own ML model.

---

## What each file does

| File | Purpose |
|------|---------|
| step1_download_data.py | Downloads/creates training images |
| step2_preprocess.py    | Cleans and prepares images for training |
| step3_train.py         | Trains DenseNet121 on your data |
| step4_evaluate.py      | Measures model quality with medical metrics |
| step5_replace_gemini.py| Drop-in replacement for ai.py |

---

## Classes the model detects

- No Finding (healthy)
- Pneumonia
- Pleural Effusion
- Cardiomegaly
- Atelectasis

---

## Important

The model output always includes a disclaimer that radiologist
review is mandatory. Never use AI predictions as final diagnosis.
