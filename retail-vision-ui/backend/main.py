import asyncio
import cv2
import numpy as np
from ultralytics import YOLOE
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import time
import logging
import base64
from PIL import Image
import os
import urllib.request
import supervision as sv
import torch
import torch.nn.functional as F

# Per-brand YOLOE class lists. The startup list must match what the frontend
# sends as text_prompt at click time — YOLOE cannot switch class counts
# post-init without tripping an internal tensor-reshape error.
RETAIL_CLASSES = ["laptop", "headphones", "glasses", "blazer", "desk", "watch",
                  "monitor", "trash can", "chair", "shirt", "running pants",
                  "running shoes", "jacket", "gloves"]
HOSPITALITY_CLASSES = ["pool", "lounge chair", "floats",
                       "beach", "ocean",
                       "golf shorts", "golfer", "golf club",
                       "food"]
BRAND_CLASSES_MAP = {
    "under-armour": RETAIL_CLASSES,
    "blend360":     RETAIL_CLASSES,
    "hyatt":        HOSPITALITY_CLASSES,
}

BRAND_VIDEO_MAP = {
    "under-armour": "../public/Under-Armour.mp4",
    "blend360":     "../public/The BLEND360 Approach.mp4",
    "hyatt":        "../public/Hyatt.mp4",
}


def get_startup_classes():
    brand_key = os.environ.get("BRAND", "").lower()
    return BRAND_CLASSES_MAP.get(brand_key, RETAIL_CLASSES)


def classes_for_brand(brand_key):
    """Return the YOLOE class list for a brand, falling back to the BRAND env
    default and then the retail list."""
    key = (brand_key or os.environ.get("BRAND", "")).lower()
    return BRAND_CLASSES_MAP.get(key, RETAIL_CLASSES)


def _class_key(classes):
    """Order-preserving cache key for a class list."""
    return tuple(classes)


def video_path_for_brand(brand_key):
    key = (brand_key or os.environ.get("BRAND", "")).lower()
    return (
        BRAND_VIDEO_MAP.get(key)
        or os.environ.get("VIDEO_PATH")
        or "../public/The BLEND360 Approach.mp4"
    )


def get_capture_for_brand(brand_key):
    key = (brand_key or os.environ.get("BRAND", "")).lower() or "blend360"
    cap = _video_caps.get(key)
    if cap is not None and cap.isOpened():
        return cap
    path = video_path_for_brand(key)
    if not os.path.exists(path):
        logger.error(f"Video file not found for brand {key}: {path}")
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for brand {key}: {path}")
        return None
    _video_caps[key] = cap
    logger.info(f"Opened video capture for brand {key}: {path}")
    return cap


# Backwards-compatible alias for any code that still references YOLOE_CLASSES
YOLOE_CLASSES = RETAIL_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MobileCLIP text embedding fix (issue #4) ---
# YOLOE's get_text_pe() can trigger a TorchScript code path where the
# positional embedding reshape fails (39424 elements vs [1, seq_len, 512]).
# We bypass this by loading MobileCLIP directly via the Python mobileclip
# package, which handles positional embedding slicing correctly.
_mobileclip_model = None
_mobileclip_tokenizer = None


def _load_mobileclip():
    """Load MobileCLIP model and tokenizer using the Python package directly."""
    global _mobileclip_model, _mobileclip_tokenizer
    if _mobileclip_model is not None:
        return True
    try:
        import mobileclip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mobileclip_model, _, _ = mobileclip.create_model_and_transforms(
            "mobileclip_b", pretrained="mobileclip_blt.pt", device=device
        )
        _mobileclip_tokenizer = mobileclip.get_tokenizer("mobileclip_b")
        logger.info("MobileCLIP loaded directly via Python package")
        return True
    except Exception as e:
        logger.warning(f"Failed to load MobileCLIP directly: {e}")
        return False


@torch.inference_mode()
def _get_text_pe_direct(texts: List[str], model):
    """Generate text positional embeddings using our directly-loaded MobileCLIP.

    Replicates the logic of YOLOEModel.get_text_pe() but uses the Python
    mobileclip package so we never hit the TorchScript positional embedding bug.
    Takes the target YOLOE model instance whose detection head builds the embeddings.
    """
    if not _load_mobileclip():
        return None
    tokens = _mobileclip_tokenizer(texts).to(
        next(_mobileclip_model.parameters()).device
    )
    txt_feats = _mobileclip_model.encode_text(tokens)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(1, len(texts), txt_feats.shape[-1])
    from ultralytics.nn.modules.head import YOLOEDetect
    head = model.model.model[-1]
    assert isinstance(head, YOLOEDetect)
    return F.normalize(head.reprta(txt_feats), dim=-1, p=2)
# --- End MobileCLIP fix ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and video on application startup."""
    # Log compute device
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} (CUDA)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("GPU detected: Apple Metal (MPS)")
    else:
        logger.info("No GPU detected, using CPU")

    # Pre-open the default brand's video capture
    default_brand = os.environ.get("BRAND", "").lower() or "blend360"
    logger.info(f"Using video for brand: {default_brand}")
    if get_capture_for_brand(default_brand) is None:
        logger.error("Failed to open default brand video on startup.")

    # Load YOLO-E v8l model with better error handling
    try:
        logger.info("Attempting to load YOLO-E v8l model...")
        if not load_yolo_e_model():
            logger.warning("Failed to initialize YOLO-E v8l model, will use fallback")
        else:
            logger.info("YOLO-E v8l model loaded successfully")
            # Warmup inference to trigger PyTorch JIT compilation
            try:
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                dummy_pil = Image.fromarray(dummy)
                yolo_e_model.predict(dummy_pil, conf=0.1, verbose=False)
                logger.info("Warmup inference complete")
            except Exception as warmup_err:
                logger.warning(f"Warmup inference failed (non-critical): {warmup_err}")
    except Exception as e:
        logger.error(f"Error during YOLO-E v8l model loading: {e}")
        logger.warning("Will use fallback models for inference")

    yield

    # Cleanup on shutdown
    for cap in _video_caps.values():
        if cap is not None:
            cap.release()
    logger.info("All video captures released")


app = FastAPI(lifespan=lifespan)

# CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if available (Docker deployment)
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/var/www/html")
if os.path.isdir(FRONTEND_DIR):
    # Mount /static for JS/CSS bundles
    static_dir = os.path.join(FRONTEND_DIR, "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="frontend-static")

# Global variables for model and video capture
_video_caps = {}  # brand_key -> cv2.VideoCapture
# Cache of YOLOE instances keyed by class tuple. Each instance has its classes
# set exactly once at load, so YOLOE never reshapes its class head at runtime.
_models = {}
yolo_e_model = None  # default model handle (set at startup) for legacy references


class ClickInferenceRequest(BaseModel):
    video_time: float
    x: int
    y: int
    frame_width: int
    frame_height: int
    text_prompt: Optional[str] = None


class DetectionResult(BaseModel):
    timestamp: float
    video_time: float
    clicked_pixel: Dict[str, int]
    detections: List[Dict[str, Any]]
    frame_base64: str
    annotated_frame_base64: str
    clicked_object: Optional[Dict[str, Any]]
    inference_type: str
    text_prompt_used: Optional[str] = None


class VideoStatus(BaseModel):
    is_loaded: bool
    total_frames: int
    fps: float
    duration: float


class TextPromptUpdateRequest(BaseModel):
    text_prompt: str


class YOLOEV8LRequest(BaseModel):
    text_prompt: str
    confidence: float = 0.1


def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        logger.info(f"Downloading {filename} from {url}...")

        # Create opener that handles redirects
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Simple download without progress bar
        urllib.request.urlretrieve(url, filename)

        logger.info(f"Download completed: {filename}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_yolo_e_v8l_model_direct() -> bool:
    """Download YOLO-E v8l model"""
    try:
        # Check if file already exists
        model_filename = "yoloe-v8l-seg.pt"
        if os.path.exists(model_filename):
            file_size = os.path.getsize(model_filename) / (1024 * 1024)
            logger.info(f"YOLO-E v8l model already exists: {model_filename} "
                       f"({file_size:.1f} MB)")
            return True

        # Download the v8l model directly
        model_url = ("https://github.com/ultralytics/assets/releases/"
                     "download/v8.3.0/yoloe-v8l-seg.pt")

        logger.info("Starting YOLO-E v8l model download...")

        if download_file(model_url, model_filename):
            logger.info("YOLO-E v8l model downloaded successfully")
            return True
        else:
            logger.error("Failed to download YOLO-E v8l model")
            return False

    except Exception as e:
        logger.error(f"Failed to download YOLO-E v8l model: {e}")
        return False


def _build_model_with_classes(classes):
    """Load a fresh YOLOE instance and set `classes` on it exactly once."""
    model_path = "yoloe-v8l-seg.pt"
    if not os.path.exists(model_path):
        if not download_yolo_e_v8l_model_direct():
            raise RuntimeError("Failed to download YOLO-E v8l model")
    model = YOLOE(model_path)
    text_pe = _get_text_pe_direct(classes, model)
    if text_pe is None:
        text_pe = model.get_text_pe(classes)
    model.set_classes(classes, text_pe)
    logger.info(f"Built YOLOE instance for classes: {classes}")
    return model


def get_model_for_brand(brand_key):
    """Return a cached YOLOE instance whose classes match the brand. Brands with
    identical class lists (blend360 + under-armour) share one instance."""
    classes = classes_for_brand(brand_key)
    key = _class_key(classes)
    model = _models.get(key)
    if model is None:
        model = _build_model_with_classes(classes)
        _models[key] = model
    return model


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert an OpenCV frame (numpy array) to a base64 encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buffer).decode('utf-8')



def get_frame_at_time(target_time: float, brand_key=None) -> np.ndarray:
    """Get frame at specific time in video"""
    cap = get_capture_for_brand(brand_key)
    if cap is None:
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    target_frame = int(target_time * fps)
    if target_frame >= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_frame = 0
        logger.info("Video looped to beginning.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    return frame if ret else None


def find_object_at_pixel(
    detections: List[Dict[str, Any]], x: int, y: int,
    frame_width: int, frame_height: int
) -> Dict[str, Any] | None:
    """Find which object (if any) contains the clicked pixel using instance
    segmentation"""
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox

        # Check if pixel is within bounding box
        if x1 <= x <= x2 and y1 <= y <= y2:
            # For segmentation, check if pixel is within mask
            if detection.get("mask"):
                try:
                    mask = np.array(detection["mask"], dtype=np.int32)
                    # Check if point is inside the mask polygon
                    if cv2.pointPolygonTest(mask, (x, y), False) >= 0:
                        return detection
                except Exception as e:
                    logger.warning(f"Mask processing error: {e}")
                    # Fallback to bounding box if mask fails
                    return detection
            else:
                # If no mask, just use bounding box
                return detection

    return None



def run_yolo_e_inference(frame: np.ndarray, clicked_x: int, clicked_y: int,
                        text_prompt: str = None) -> Dict[str, Any]:
    """Run YOLO-E inference - simple and direct like your reference code"""
    try:
        if yolo_e_model is None:
            return {"error": "No YOLO-E model available"}

        timings: Dict[str, float] = {}

        # Convert frame to PIL Image for YOLOE
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        timings["pil_convert_ms"] = (time.perf_counter() - t0) * 1000

        # Parse text prompt into classes and set them on the model
        # _set_classes_cached is the critical-path step: a cache miss here
        # rebuilds MobileCLIP embeddings (~2-3s on CPU). Time it to detect that.
        t0 = time.perf_counter()
        if text_prompt:
            try:
                prompt_classes = [cls.strip() for cls in text_prompt.split(',')
                                if cls.strip()]
                if prompt_classes:
                    _set_classes_cached(prompt_classes)
                else:
                    logger.warning("No valid classes found in text prompt")
            except Exception as e:
                logger.warning(f"Could not set text prompts: {e}")
        timings["set_classes_ms"] = (time.perf_counter() - t0) * 1000

        # Resize for faster inference (YOLO resizes internally anyway)
        t0 = time.perf_counter()
        INFER_SIZE = 640
        orig_h, orig_w = frame.shape[:2]
        scale = min(INFER_SIZE / orig_w, INFER_SIZE / orig_h)
        if scale < 1.0:
            infer_w, infer_h = int(orig_w * scale), int(orig_h * scale)
            infer_image = pil_image.resize((infer_w, infer_h), Image.BILINEAR)
        else:
            scale = 1.0
            infer_image = pil_image
        timings["resize_ms"] = (time.perf_counter() - t0) * 1000

        # Run YOLO-E inference on resized image
        t0 = time.perf_counter()
        results = yolo_e_model.predict(infer_image, conf=0.1, verbose=False)
        timings["predict_ms"] = (time.perf_counter() - t0) * 1000

        if not results or len(results) == 0:
            logger.info(f"YOLO-E timing (no results) {timings}")
            return {
                "detections": [],
                "total_objects": 0,
                "annotated_frame": frame.copy(),
                "clicked_object": None,
                "text_prompt_used": text_prompt if text_prompt else "default_classes"
            }

        # Process results from YOLOE (mask extraction + per-detection annotation)
        t0 = time.perf_counter()
        result = results[0]
        detections = []
        annotated_frame = frame.copy()

        # Check if we have detections
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for i, box in enumerate(boxes):
                try:
                    # Extract bounding box coordinates and scale back to original frame
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = bx1 / scale, by1 / scale, bx2 / scale, by2 / scale
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else 0

                    # Get class name from text prompt classes or model names
                    class_name = "unknown"
                    if text_prompt:
                        prompt_classes = [cls.strip() for cls in text_prompt.split(',')
                                        if cls.strip()]
                        if class_id < len(prompt_classes):
                            class_name = prompt_classes[class_id]
                        else:
                            class_name = f"object_{class_id}"
                    elif hasattr(yolo_e_model, 'names') and yolo_e_model.names:
                        if class_id < len(yolo_e_model.names):
                            class_name = yolo_e_model.names[class_id]
                        else:
                            class_name = f"object_{class_id}"
                    else:
                        class_name = f"object_{i}"

                    # Extract segmentation mask if available, scale to original frame
                    mask = None
                    if masks is not None and i < len(masks):
                        try:
                            if hasattr(masks, 'xy'):
                                mask = (np.array(masks.xy[i]) / scale).tolist()
                            elif hasattr(masks, 'data'):
                                mask_data = masks.data[i].cpu().numpy()
                                contours, _ = cv2.findContours(
                                    mask_data.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                )
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    mask = (largest_contour.squeeze() / scale).tolist()
                        except Exception as e:
                            logger.warning(f"Could not extract mask for object {i}: {e}")

                    # Create detection object
                    detection = {
                        "id": i,
                        "class": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "mask": mask
                    }

                    detections.append(detection)

                    # Draw visualization using our existing OpenCV functions
                    try:
                        # Draw bounding box
                        cv2.rectangle(
                            annotated_frame, (int(x1), int(y1)),
                            (int(x2), int(y2)), (0, 255, 255), 2
                        )

                        # Draw mask if available
                        if mask is not None:
                            mask_array = np.array(mask, dtype=np.int32)
                            # Create semi-transparent mask overlay
                            mask_overlay = np.zeros_like(frame)
                            cv2.fillPoly(mask_overlay, [mask_array], (0, 255, 255))
                            # Blend with frame
                            annotated_frame = cv2.addWeighted(
                                annotated_frame, 0.7,
                                mask_overlay, 0.3, 0
                            )
                            # Draw mask boundary
                            cv2.polylines(
                                annotated_frame, [mask_array], True,
                                (0, 255, 255), 2
                            )

                        # Add label
                        label = f"{class_name} {confidence:.2f}"
                        label_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )[0]
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), (0, 255, 255), -1
                        )
                        cv2.putText(
                            annotated_frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                        )

                    except Exception as e:
                        logger.warning(f"Could not draw visualization for "
                                     f"object {i}: {e}")

                except Exception as e:
                    logger.warning(f"Error processing object {i}: {e}")
                    continue

        # Find which object (if any) contains the clicked pixel
        clicked_object = find_object_at_pixel(
            detections, clicked_x, clicked_y,
            frame.shape[1], frame.shape[0]
        )

        # Highlight the clicked pixel
        cv2.circle(annotated_frame, (clicked_x, clicked_y), 5, (255, 0, 255),
                   -1)
        timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000
        logger.info(
            f"YOLO-E timing dets={len(detections)} "
            f"set_classes={timings['set_classes_ms']:.0f}ms "
            f"predict={timings['predict_ms']:.0f}ms "
            f"postprocess={timings['postprocess_ms']:.0f}ms "
            f"pil={timings['pil_convert_ms']:.0f}ms "
            f"resize={timings['resize_ms']:.0f}ms"
        )

        return {
            "detections": detections,
            "total_objects": len(detections),
            "annotated_frame": annotated_frame,
            "clicked_object": clicked_object,
            "text_prompt_used": text_prompt if text_prompt else "default_classes"
        }

    except Exception as e:
        logger.error(f"YOLO-E inference error: {e}")
        return {"error": str(e)}


def run_yolo_e_v8l_inference(
    frame: np.ndarray, text_prompt: str, confidence: float = 0.1
) -> Dict[str, Any]:
    """Run YOLO-E v8l inference using the clean approach from reference code"""
    try:
        if yolo_e_model is None:
            return {"error": "No YOLO-E model available"}

        # Convert frame to PIL Image for YOLOE
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Parse text prompt into classes
        prompt_classes = [cls.strip() for cls in text_prompt.split(',')
                        if cls.strip()]

        if not prompt_classes:
            return {"error": "No valid classes found in text prompt"}

        # Set classes with cached text embeddings
        _set_classes_cached(prompt_classes)

        # Resize for faster inference
        INFER_SIZE = 640
        orig_h, orig_w = frame.shape[:2]
        scale = min(INFER_SIZE / orig_w, INFER_SIZE / orig_h)
        if scale < 1.0:
            infer_w, infer_h = int(orig_w * scale), int(orig_h * scale)
            infer_image = pil_image.resize((infer_w, infer_h), Image.BILINEAR)
        else:
            scale = 1.0
            infer_image = pil_image

        # Run YOLO-E inference on resized image
        results = yolo_e_model.predict(infer_image, conf=confidence, verbose=False)

        if not results or len(results) == 0:
            return {
                "detections": [],
                "total_objects": 0,
                "annotated_frame": frame.copy(),
                "text_prompt_used": text_prompt
            }

        # Convert results to supervision format for better mask handling
        try:
            detections = sv.Detections.from_ultralytics(results[0])

            # Create annotated frame using supervision
            annotated_frame = frame.copy()

            # Convert PIL back to OpenCV for annotation
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame,
                                                       cv2.COLOR_BGR2RGB))

            # Apply mask annotations
            annotated_pil = sv.MaskAnnotator().annotate(
                scene=annotated_pil, detections=detections
            )

            # Apply label annotations
            annotated_pil = sv.LabelAnnotator().annotate(
                scene=annotated_pil, detections=detections
            )

            # Convert back to OpenCV format
            annotated_frame = cv2.cvtColor(np.array(annotated_pil),
                                         cv2.COLOR_RGB2BGR)

            # Convert detections to our format
            detection_list = []
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                for i in range(len(detections.xyxy)):
                    detection = {
                        "id": i,
                        "class": i,
                        "class_name": prompt_classes[i] if i < len(prompt_classes)
                            else f"object_{i}",
                        "confidence": detections.confidence[i] if
                            detections.confidence is not None else 0.0,
                        "bbox": detections.xyxy[i].tolist(),
                        "mask": detections.mask[i].tolist() if
                            detections.mask is not None else None
                    }
                    detection_list.append(detection)

            return {
                "detections": detection_list,
                "total_objects": len(detection_list),
                "annotated_frame": annotated_frame,
                "text_prompt_used": text_prompt
            }

        except Exception as sv_error:
            logger.warning(f"Supervision annotation failed, using fallback: "
                         f"{sv_error}")
            # Fallback to basic OpenCV annotation
            return run_yolo_e_inference(frame, 0, 0, text_prompt)

    except Exception as e:
        logger.error(f"YOLO-E v8l inference error: {e}")
        return {"error": str(e)}




@app.get("/api/health")
async def health_check():
    return {"message": "Retail Vision Backend API"}


@app.get("/api/video-status", response_model=VideoStatus)
async def get_video_status():
    """Get the current status of the loaded video."""
    cap = get_capture_for_brand(None)
    if cap is None:
        return VideoStatus(is_loaded=False, total_frames=0, fps=0, duration=0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    return VideoStatus(
        is_loaded=True,
        total_frames=total_frames,
        fps=fps,
        duration=duration
    )


@app.post("/api/inference/yolo-e")
async def get_yolo_e_inference(request: ClickInferenceRequest):
    """Get YOLO-E inference results for a specific pixel click on a video frame"""
    try:
        req_start = time.perf_counter()

        # Get frame at the specified video time
        t0 = time.perf_counter()
        frame = get_frame_at_time(request.video_time)
        frame_fetch_ms = (time.perf_counter() - t0) * 1000
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        logger.info(f"Click coordinates: ({request.x}, {request.y}) on frame "
                   f"{frame.shape[1]}x{frame.shape[0]}")

        # Run YOLO-E inference in thread pool to avoid blocking the event loop
        t0 = time.perf_counter()
        inference_result = await asyncio.to_thread(
            run_yolo_e_inference, frame, request.x, request.y, request.text_prompt
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        if "error" in inference_result:
            raise HTTPException(
                status_code=500, detail=inference_result["error"]
            )

        # Convert frames to base64
        t0 = time.perf_counter()
        frame_base64 = frame_to_base64(frame)
        annotated_frame_base64 = frame_to_base64(
            inference_result.get("annotated_frame", frame)
        )
        b64_ms = (time.perf_counter() - t0) * 1000

        # Create result
        result = DetectionResult(
            timestamp=time.time(),
            video_time=request.video_time,
            clicked_pixel={"x": request.x, "y": request.y},
            detections=inference_result.get("detections", []),
            frame_base64=frame_base64,
            annotated_frame_base64=annotated_frame_base64,
            clicked_object=inference_result.get("clicked_object"),
            inference_type="YOLO-E",
            text_prompt_used=inference_result.get("text_prompt_used")
        )

        total_ms = (time.perf_counter() - req_start) * 1000
        logger.info(
            f"/api/inference/yolo-e total={total_ms:.0f}ms "
            f"frame_fetch={frame_fetch_ms:.0f}ms "
            f"inference={inference_ms:.0f}ms "
            f"base64={b64_ms:.0f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"YOLO-E inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/yolo-e/update-prompt")
async def update_yolo_e_prompt(request: TextPromptUpdateRequest):
    """Update YOLO-E text prompts for object detection"""
    try:
        if yolo_e_model is None:
            raise HTTPException(
                status_code=400, detail="YOLO-E model not loaded"
            )

        if not hasattr(yolo_e_model, 'set_classes'):
            raise HTTPException(
                status_code=400, detail="YOLO-E model doesn't support text prompts"
            )

        # Parse text prompt into classes
        prompt_classes = [cls.strip() for cls in request.text_prompt.split(',')
                        if cls.strip()]

        if not prompt_classes:
            raise HTTPException(
                status_code=400, detail="No valid classes found in text prompt"
            )

        # Update the model's text prompts (invalidates cache for new prompt)
        try:
            _set_classes_cached(prompt_classes)
            logger.info(f"YOLO-E text prompts updated: {prompt_classes}")
        except Exception as e:
            logger.error(f"Failed to update YOLO-E text prompts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "message": "YOLO-E text prompts updated successfully",
            "classes": prompt_classes,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to update YOLO-E text prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/yolo-e/current-prompt")
async def get_current_yolo_e_prompt():
    """Get current YOLO-E text prompts"""
    try:
        if yolo_e_model is None:
            raise HTTPException(
                status_code=400, detail="YOLO-E model not loaded"
            )

        if hasattr(yolo_e_model, 'names') and yolo_e_model.names:
            current_classes = list(yolo_e_model.names.values())
            return {
                "current_prompt": ", ".join(current_classes),
                "classes": current_classes,
                "timestamp": time.time()
            }
        else:
            return {
                "current_prompt": "limited_retail_classes",
                "classes": YOLOE_CLASSES,
                "timestamp": time.time()
            }

    except Exception as e:
        logger.error(f"Failed to get current YOLO-E text prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inference/yolo-e-v8l")
async def get_yolo_e_v8l_inference(request: YOLOEV8LRequest):
    """Get YOLO-E v8l inference results for a video frame with text prompts"""
    try:
        # Get current frame from video (or use a default frame)
        cap = get_capture_for_brand(None)
        if cap is not None:
            # Get current frame position
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame <= 0:
                # If at beginning, get first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = cap.read()
            if not ret:
                # If no frame available, create a test frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No video frame available", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Create a test frame if no video is loaded
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video loaded", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Run YOLO-E v8l inference in thread pool to avoid blocking the event loop
        inference_result = await asyncio.to_thread(
            run_yolo_e_v8l_inference, frame, request.text_prompt, request.confidence
        )

        if "error" in inference_result:
            raise HTTPException(
                status_code=500, detail=inference_result["error"]
            )

        # Convert frames to base64
        frame_base64 = frame_to_base64(frame)
        annotated_frame_base64 = frame_to_base64(
            inference_result.get("annotated_frame", frame)
        )

        # Create result
        result = DetectionResult(
            timestamp=time.time(),
            video_time=0.0,  # Not applicable for this endpoint
            clicked_pixel={"x": 0, "y": 0},  # Not applicable for this endpoint
            detections=inference_result.get("detections", []),
            frame_base64=frame_base64,
            annotated_frame_base64=annotated_frame_base64,
            clicked_object=None,  # Not applicable for this endpoint
            inference_type="YOLO-E-v8l",
            text_prompt_used=inference_result.get("text_prompt_used")
        )

        logger.info(
            f"YOLO-E v8l inference with prompt '{request.text_prompt}' - "
            f"{len(inference_result.get('detections', []))} objects detected"
        )

        return result

    except Exception as e:
        logger.error(f"YOLO-E v8l inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/yolo-e-v8l/status")
async def get_yolo_e_v8l_status():
    """Get the status of the YOLO-E v8l model"""
    try:
        if yolo_e_model is None:
            return {
                "model_loaded": False,
                "model_name": "YOLO-E v8l",
                "status": "Not loaded",
                "timestamp": time.time()
            }

        # Get model info
        model_info = {
            "model_loaded": True,
            "model_name": "YOLO-E v8l",
            "status": "Loaded and ready",
            "timestamp": time.time()
        }

        # Try to get model properties
        try:
            if hasattr(yolo_e_model, 'names') and yolo_e_model.names:
                model_info["available_classes"] = list(
                    yolo_e_model.names.values())
            else:
                model_info["available_classes"] = []
        except Exception as e:
            logger.warning(f"Could not get model classes: {e}")
            model_info["available_classes"] = []

        return model_info

    except Exception as e:
        logger.error(f"Failed to get YOLO-E v8l status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve video files with Range request support (enables seeking in browser)
# Serve video files with Range request support (enables seeking in browser)
VIDEO_DIR = os.environ.get("VIDEO_DIR", "../public")
if not os.path.isdir(VIDEO_DIR):
    VIDEO_DIR = "/app/public"


@app.get("/videos/{video_name:path}")
async def serve_video(video_name: str, request: Request):
    video_file = os.path.join(VIDEO_DIR, video_name)
    if not os.path.isfile(video_file):
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = os.path.getsize(video_file)
    range_header = request.headers.get("range")

    if range_header:
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        range_start = int(parts[0])
        range_end = int(parts[1]) if parts[1] else file_size - 1
        content_length = range_end - range_start + 1

        def iter_file():
            with open(video_file, "rb") as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
            },
        )

    def iter_full_file():
        with open(video_file, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_full_file(),
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        },
    )


# Serve frontend as SPA with html=True (client-side routing)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend-spa")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
