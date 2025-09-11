#!/usr/bin/env python3
"""
Retail Vision Backend API
FastAPI application for object detection and segmentation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import logging
import os
import cv2
import base64
from PIL import Image
from ultralytics import YOLO
import time
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
yolo_e_model = None
video_cap = None

# Model and video paths
MODEL_PATH = "../../yoloe-v8l-seg.pt"
VIDEO_PATH = "../../static/videos/The BLEND360 Approach.mp4"

# Pydantic models for request/response


class InferenceRequest(BaseModel):
    video_time: float
    x: int
    y: int
    frame_width: int
    frame_height: int
    text_prompt: Optional[str] = (
        "laptop, headphones, glasses, blazer, desk, watch, monitor, "
        "trash can, chair, shirt"
    )
    confidence: Optional[float] = 0.1


class Detection(BaseModel):
    id: int
    class_: int
    class_name: str
    confidence: float
    bbox: List[float]
    mask: Optional[List[List[int]]] = None


class InferenceResult(BaseModel):
    timestamp: float
    video_time: float
    clicked_pixel: Dict[str, int]
    detections: List[Detection]
    frame_base64: str
    annotated_frame_base64: str
    clicked_object: Optional[Detection] = None
    inference_type: str
    text_prompt_used: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="Retail Vision API",
    description="Computer vision API for object detection and segmentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading function
def load_models():
    """Load YOLO-E model and video capture"""
    global yolo_e_model, video_cap
    
    try:
        # Load YOLO-E model
        if os.path.exists(MODEL_PATH):
            yolo_e_model = YOLO(MODEL_PATH)
            logger.info(f"Loaded YOLO-E model from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
            
        # Load video
        if os.path.exists(VIDEO_PATH):
            video_cap = cv2.VideoCapture(VIDEO_PATH)
            if not video_cap.isOpened():
                logger.error(f"Could not open video: {VIDEO_PATH}")
                return False
            logger.info(f"Loaded video from {VIDEO_PATH}")
        else:
            logger.error(f"Video file not found: {VIDEO_PATH}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    logger.info("Starting up Retail Vision Backend...")
    if load_models():
        logger.info("Models loaded successfully")
    else:
        logger.error("Failed to load models")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global video_cap
    if video_cap:
        video_cap.release()
    logger.info("Shutdown complete")

# Detect if running in Docker or locally
IS_DOCKER = os.path.exists("/var/www/html/static")

if IS_DOCKER:
    # Docker environment - serve built frontend
    app.mount("/static", StaticFiles(directory="/var/www/html/static"), name="static")
    
    @app.get("/")
    async def root():
        """Root endpoint - serve frontend"""
        return FileResponse("/var/www/html/index.html")
else:
    # Local development - just serve API, frontend runs separately
    @app.get("/")
    async def root():
        """Root endpoint - API only in local dev"""
        return {"message": "Retail Vision Backend API - Frontend runs on port 3000"}

@app.get("/api")
async def api_info():
    """API info endpoint"""
    return {"message": "Retail Vision Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Retail Vision API is running"}

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "models": [
            {
                "name": "YOLO-E v8l-seg",
                "type": "segmentation",
                "status": "available"
            }
        ]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict objects in uploaded image"""
    try:
        # For now, return a mock response
        return {
            "predictions": [
                {
                    "class": "mock_object",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 200],
                    "mask": "base64_encoded_mask"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/yolo-e", response_model=InferenceResult)
async def yolo_e_inference(request: InferenceRequest):
    """Run YOLO-E inference on video frame at specified time and position"""
    global yolo_e_model, video_cap

    if not yolo_e_model or not video_cap:
        raise HTTPException(
            status_code=500, detail="Model or video not loaded"
        )

    try:
        # Set video to the specified time
        video_cap.set(cv2.CAP_PROP_POS_MSEC, request.video_time * 1000)
        ret, frame = video_cap.read()

        if not ret:
            raise HTTPException(
                status_code=400, detail="Could not read frame at specified time"
            )

        # Convert frame to PIL Image for YOLO-E
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Note: YOLO-E text prompts are handled differently
        # For now, we'll use the default classes

        # Run inference
        results = yolo_e_model(pil_image, conf=request.confidence)

        # Process results
        detections = []
        clicked_object = None

        for i, result in enumerate(results):
            if result.masks is not None:
                for j, (box, mask, conf, cls) in enumerate(zip(
                    result.boxes.xyxy, result.masks.data,
                    result.boxes.conf, result.boxes.cls
                )):
                    class_name = yolo_e_model.names[int(cls)]

                    detection = Detection(
                        id=j,
                        class_=int(cls),
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=box.tolist(),
                        mask=mask.tolist() if mask is not None else None
                    )
                    detections.append(detection)

                    # Check if clicked pixel is within this detection's bbox
                    if (request.x >= box[0] and request.x <= box[2] and
                            request.y >= box[1] and request.y <= box[3]):
                        clicked_object = detection

        # Create annotated frame
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{detection.class_name}: {detection.confidence:.2f}",
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Convert frames to base64
        _, frame_buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(frame_buffer).decode('utf-8')

        _, annotated_buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_frame_base64 = base64.b64encode(
            annotated_buffer
        ).decode('utf-8')

        # Create response
        result = InferenceResult(
            timestamp=time.time(),
            video_time=request.video_time,
            clicked_pixel={"x": request.x, "y": request.y},
            detections=detections,
            frame_base64=frame_base64,
            annotated_frame_base64=annotated_frame_base64,
            clicked_object=clicked_object,
            inference_type="YOLO-E",
            text_prompt_used=request.text_prompt
        )

        return result

    except Exception as e:
        logger.error(f"YOLO-E inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the main index.html for all non-API routes (must be last)
if IS_DOCKER:
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes"""
        # Serve index.html for all other routes (SPA routing)
        return FileResponse("/var/www/html/index.html")

if __name__ == "__main__":
    logger.info("Starting Retail Vision Backend...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )