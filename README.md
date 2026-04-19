# ANPR Vision 🔍 — Indian Number Plate Recognition

> A full-stack Machine Learning application for **Automatic Number Plate Recognition (ANPR)** of Indian vehicles.  
> Upload a car image *or* stream live webcam footage to detect and extract license plate text in real time.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔍 Plate Detection | OpenCV Haar Cascade (`haarcascade_russian_plate_number.xml`) |
| 🔤 OCR Engine | EasyOCR — no Tesseract binary required |
| 🇮🇳 Indian plate patterns | Regex post-processing for MH, DL, KA, BH series, etc. |
| 📁 Image upload | Drag-and-drop + file browser |
| 📷 Live webcam | WebSocket-based frame streaming (~1.25 fps) |
| ⚡ Backend | FastAPI + Uvicorn |
| 🎨 Frontend | React + Vite, dark glassmorphism UI |
| 🧠 Deep model (optional) | InceptionResNetV2 bounding-box regressor (TensorFlow) |
| 📦 Dataset ingestion | Automated Kaggle download + Pascal-VOC XML parser |

---

## 📁 Project Structure

```
mldl_work/
├── .agents/
│   └── skills/
│       ├── plate_detector.py    ← Core ML: detection + OCR
│       ├── data_ingestion.py    ← Kaggle download + annotation parser
│       └── training.py          ← InceptionResNetV2 training script
├── backend/
│   ├── main.py                  ← FastAPI app (REST + WebSocket)
│   ├── requirements.txt
│   └── .venv/                   ← Python virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Main React component
│   │   ├── main.jsx
│   │   └── index.css            ← Dark-mode design system
│   ├── index.html
│   ├── .env                     ← VITE_API_URL
│   └── package.json
├── data/
│   ├── raw/                     ← Downloaded Kaggle datasets
│   └── processed/               ← JSON manifests
├── models/                      ← Saved model weights
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 – 3.13 |
| Node.js | 18 + |
| npm | 9 + |

---

### 1. Clone (or enter your workspace)

```bash
cd mldl_work
```

### 2. Backend Setup

```bash
cd backend

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate            # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

The API will be live at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies (first time only)
npm install

# Start the dev server
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 API Reference

### `GET /health`
Liveness check.
```json
{ "status": "ok", "service": "ANPR API v1.0" }
```

### `POST /detect`
Upload an image, get plate detections.

**Request:** `multipart/form-data` with field `file` (image/*)

**Response:**
```json
{
  "plates": [
    {
      "text": "MH12AB1234",
      "confidence": 0.87,
      "bbox": [120, 340, 200, 60],
      "crop_image": "data:image/jpeg;base64,..."
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "total_found": 1
}
```

### `WS /ws/detect`
WebSocket endpoint for live camera frames.  
**Send:** raw JPEG bytes  
**Receive:** same JSON structure as `/detect`

---

## 🗃️ Dataset Setup (Optional — for Training)

### Configure Kaggle API

1. Go to https://www.kaggle.com/account → Create New Token
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. On Windows, set permissions: `icacls "%USERPROFILE%\.kaggle\kaggle.json" /inheritance:r /grant:r "%username%:(R)"`

### Download Datasets

```bash
cd backend
.\.venv\Scripts\Activate.ps1
python -c "
import sys
sys.path.insert(0, '..')
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('di', '../.agents/skills/data_ingestion.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ingest_all()
"
```

This downloads:
- `alihassanml/car-number-plate` → `data/raw/car-number-plate/`
- `suprabhosaha/indian-vehicle-number-plate-detection-dataset` → `data/raw/indian-vehicle-number-plate-detection-dataset/`

And creates JSON manifests in `data/processed/`.

---

## 🧠 Training the Deep Model (Optional)

```bash
cd backend
.\.venv\Scripts\Activate.ps1
pip install tensorflow scikit-learn  # extra deps

python -c "
import sys; sys.path.insert(0, '..')
import importlib.util as ilu
spec = ilu.spec_from_file_location('training', '../.agents/skills/training.py')
m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
m.train()
"
```

The trained model is saved to `models/anpr_model.keras`. Training runs for up to 50 epochs with early stopping.

---

## 🔧 Configuration

| File | Purpose |
|------|---------|
| `backend/.venv` | Python virtual environment |
| `frontend/.env` | `VITE_API_URL` — backend URL |
| `.env.example` | Template for environment variables |
| `.gitignore` | Excludes models, datasets, node_modules |

---

## 🇮🇳 Indian Plate Formats Supported

| Series | Example | Pattern |
|--------|---------|---------|
| Standard | `MH12AB1234` | `XX99XX9999` |
| Old format | `DL01CD5678` | `XX99XX9999` |
| BH series | `22BH1234AB` | `99BH9999XX` |
| Single letter | `KA05E1234` | `XX99X9999` |

---

## 📦 Dependencies Summary

### Backend
- `fastapi` + `uvicorn` — ASGI web framework
- `easyocr` — OCR engine (PyTorch-based, no Tesseract needed)
- `opencv-python-headless` — Haar Cascade plate detection
- `numpy`, `Pillow` — image utilities
- `tensorflow` *(optional)* — deep model training

### Frontend
- `react` + `vite` — UI framework & bundler
- Vanilla CSS — no Tailwind; custom dark design system

---

## 📄 License

MIT — see [`LICENSE`](./LICENSE)

---

## 🙏 Acknowledgements

- Reference notebook: [alihassanml/Car-Number-Plates-Detection](https://github.com/alihassanml/Car-Number-Plates-Detection)
- Dataset: [alihassanml/car-number-plate](https://www.kaggle.com/datasets/alihassanml/car-number-plate)
- Dataset: [suprabhosaha/indian-vehicle-number-plate-detection-dataset](https://www.kaggle.com/datasets/suprabhosaha/indian-vehicle-number-plate-detection-dataset)
