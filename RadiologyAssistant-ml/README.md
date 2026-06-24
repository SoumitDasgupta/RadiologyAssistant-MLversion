# 🩻 AI Radiology Assistant

An AI-powered radiology image analysis tool built with a modern dark medical UI. Upload chest X-rays, MRIs, or any radiology image and get instant structured findings, clinical impressions, and recommendations — powered by **Google Gemini AI**.

![AI Radiology Assistant](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-blue?style=flat-square&logo=google)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)
![Database](https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- **AI Image Analysis** — Upload JPEG, PNG, or DICOM images and get structured radiology findings in seconds
- **Gemini 2.5 Flash Vision** — Google's latest multimodal AI model analyzes the image and returns confidence score, findings list, clinical impression, and recommendations
- **Image Viewer** — Brightness, contrast, sharpness controls + window presets (Bone, Lung, Soft Tissue, Vascular, Invert)
- **Annotation Overlay** — AI-detected findings are marked directly on the image with labeled bubbles
- **Report Generation** — Download a formatted `.txt` radiology report with one click
- **Analysis History** — Every scan is saved to a local SQLite database and retrievable via REST API
- **Dark Medical UI** — Clean dark theme designed to match clinical environments, with a light mode toggle

---

## 🏗 Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Frontend  | Vanilla HTML, CSS, JavaScript     |
| Backend   | Python, FastAPI, Uvicorn          |
| AI Model  | Google Gemini 2.5 Flash (Vision), Groq(Compound) |
| Database  | SQLite (via Python sqlite3)       |
| Image     | Pillow (PIL)                      |
| Reports   | ReportLab                         |

---

## 📁 Project Structure

```
Radiology-AI/
├── backend/
│   ├── app.py            # FastAPI routes (/analyze, /history, /report)
│   ├── ai.py             # Gemini Vision integration
│   ├── database.py       # SQLite setup and connection
│   ├── models.py         # Data models
│   ├── report.py         # Report generation logic
│   ├── requirements.txt  # Python dependencies
│   └── .env              # API keys (not committed)
├── frontend/
│   ├── index.html        # Full single-page app
│   ├── style.css         # Dark medical UI styles
│   └── app.js            # All frontend logic
├── uploads/              # Uploaded images (auto-created)
├── reports/              # Generated reports
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/radiology-ai.git
cd radiology-ai
```

### 2. Set up the backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file inside the `backend/` folder:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get a free API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Start the backend server

```bash
uvicorn app:app --reload
```

The API will be running at `http://127.0.0.1:8000`

### 5. Open the frontend

Open `frontend/index.html` directly in your browser — no build step needed.

---

## 🔌 API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Health check                       |
| POST   | `/analyze`        | Upload image → returns AI analysis |
| GET    | `/history`        | List all past analyses             |
| GET    | `/report/{id}`    | Fetch a specific report by ID      |

### Example `/analyze` response

```json
{
  "filename": "chest_xray_001.jpg",
  "analysis": {
    "findings": "Small pulmonary nodule in the left upper lobe...",
    "findings_list": [
      { "title": "Pulmonary Nodule", "details": "Left upper lobe · 8mm · Round · Well-defined margins" },
      { "title": "Mild Cardiomegaly", "details": "Cardiothoracic ratio 0.52 · Borderline enlarged" }
    ],
    "impression": "The chest radiograph demonstrates a small, well-defined pulmonary nodule in the left upper lobe measuring approximately 8mm. Mild cardiomegaly is noted.",
    "confidence": 92,
    "recommendation": "3-month follow-up CT scan recommended.",
    "recommendations_list": [
      "3-month follow-up CT scan",
      "Cardiology consultation advised",
      "PET scan if nodule grows >6mm"
    ]
  }
}
```

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. All AI-generated findings must be reviewed and verified by a qualified, licensed radiologist before any clinical use. Do not use this tool as a substitute for professional medical diagnosis.

---

## 📄 License

MIT License — free to use, modify, and distribute.
