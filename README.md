# Fashion Recommender

An AI-powered fashion recommendation system that uses **CLIP** for visual similarity search and **BLIP** for natural-language image captioning.

## Features

- **Multi-modal search** — query by text, image, or both with adjustable weights
- **Multi-dataset fusion** — unified FAISS index across DeepFashion2, eBay, and Amazon fashion datasets
- **Personalized ranking** — preference tracking with accept/reject feedback
- **Explainability** — CLIP attribute probes + BLIP captions explain why items are recommended
- **Preference drift detection** — automatically adapts when your taste shifts
- **Stratified retrieval** — per-source sub-indexes ensure dataset diversity in results
- **User authentication** — role-based access with admin dashboard

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| ML Models | CLIP ViT-B/32, BLIP-base (Hugging Face Transformers) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Frontend | Vanilla HTML/CSS/JS |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure dataset paths

Edit the `_BASE` and `DATASET_CONFIGS` variables in `recommender.py` to point to your local dataset directories.

### 3. Run

```bash
python app.py
```

The server starts on `http://localhost:5000`. On first launch it will:
1. Load CLIP and encode all dataset images (cached after first run)
2. Build FAISS indexes
3. Download and load the BLIP captioning model (~1GB)

### Default admin credentials

- **Username:** `admin`
- **Password:** `admin`

> ⚠️ Change these and the Flask `secret_key` before any production use.

## Project Structure

```
fashion-recommender/
├── app.py              # Flask routes & API endpoints
├── recommender.py      # Core ML pipeline (CLIP, FAISS, BLIP, reranking)
├── auth.py             # User authentication & preference persistence
├── requirements.txt    # Python dependencies
├── static/
│   ├── app.js          # Frontend search UI logic
│   ├── style.css       # Main application styles
│   └── auth.css        # Login/auth page styles
└── templates/
    ├── index.html       # Main search interface
    ├── login.html       # Landing login page
    ├── login_user.html  # User login/register
    ├── login_admin.html # Admin login
    ├── admin.html       # Admin dashboard
    └── admin_user.html  # Admin user detail view
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Health check & model readiness |
| POST | `/api/search` | Search by text/image (multipart form) |
| POST | `/api/accept` | Mark item as liked |
| POST | `/api/reject` | Mark item as disliked |
| POST | `/api/reset` | Clear preference history |

## License

MIT
