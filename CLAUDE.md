# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retail Vision is a computer vision application that enables click-to-purchase on objects in videos. Users watch video content, click on detected objects (clothing, accessories), and add them to a shopping cart. It uses YOLO-E segmentation models with MobileCLIP for text-prompted object detection.

## Architecture

**Two-service monorepo:**

1. **Backend** (`retail-vision-ui/backend/`) — FastAPI (Python 3.11+) server running YOLO-E v8l inference
   - `main.py` is the single API file containing all endpoints, model loading, and video processing
   - Uses `ultralytics.YOLOE` for object detection/segmentation and `supervision` for annotation
   - Models loaded at startup via lifespan context manager: `yoloe-v8l-seg.pt` (~107MB) + `mobileclip_blt.pt`/`.ts`
   - CORS configured for `http://localhost:3000`

2. **Frontend** (`retail-vision-ui/`) — React 18 + TypeScript with Material-UI (Create React App)
   - `src/App.tsx` — Main component, manages cart state (`CartItem[]`), theme, layout
   - `src/components/VideoPlayer.tsx` — Interactive video player with click-to-detect
   - `src/components/InferencePanel.tsx` — Displays detection results from backend
   - `src/components/ShoppingCart.tsx` — Shopping cart with size/color selectors

**Data flow:** User clicks video frame → frontend sends `POST /api/inference` with coordinates + timestamp → backend extracts frame, runs YOLO-E inference → returns detections with masks + annotated frame as base64 → frontend displays results and allows add-to-cart.

## Common Commands

### Backend
```bash
cd retail-vision-ui/backend
source venv/bin/activate
python run_backend.py                          # Start with auto-reload (port 8000)
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Alternative
```

### Frontend
```bash
cd retail-vision-ui
npm start       # Dev server (port 3000)
npm run build   # Production build
npm test        # Jest tests
```

### Both services at once
```bash
./quick_launch.sh   # Kills ports 3000/8000, starts both services in background
```

### Docker
```bash
docker build -t retail-vision .    # Multi-stage build (frontend + backend)
docker run -p 8000:8000 retail-vision
```

## Key API Endpoints

- `GET /` — Health check
- `GET /api/status` — Video and model status
- `POST /api/inference` — Click-based inference (body: `{video_time, x, y, frame_width, frame_height, text_prompt?}`)
- `GET /api/inference/latest` — Latest inference result
- `POST /api/inference/yolo-e-v8l` — Direct YOLO-E v8l inference (body: `{text_prompt, confidence?}`)
- `GET /api/yolo-e-v8l/status` — Model status

## Important Details

- The video file path is hardcoded in `main.py`: `../public/Under-Armour.mp4` (relative to backend dir)
- Default YOLOE detection classes are defined in `YOLOE_CLASSES` at the top of `main.py`
- Model files (`*.pt`, `*.ts`) are large and stored in `retail-vision-ui/backend/` — they are gitignored
- Backend venvs exist at both `./venv/` (root, Python 3.12) and `retail-vision-ui/backend/venv/` (Python 3.11) — use the backend one for running the server
- YOLO-E requires special pip installs from GitHub (see `requirements.txt` for the `git+` URLs)
- Frontend proxies API calls to backend at `http://localhost:8000`
