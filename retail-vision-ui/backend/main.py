import cv2
import numpy as np
from ultralytics import YOLOE
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Single source of truth for YOLOE classes
YOLOE_CLASSES = ["laptop", "headphones", "glasses", "blazer", "desk", "watch",
                 "monitor", "trash can", "chair", "shirt"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and video on application startup."""
    video_path = "../../static/videos/The BLEND360 Approach.mp4"

    if not load_video(video_path):
        logger.error("Failed to load video on startup.")

    # Load YOLO-E v8l model with better error handling
    try:
        logger.info("Attempting to load YOLO-E v8l model...")
        if not load_yolo_e_model():
            logger.warning("Failed to initialize YOLO-E v8l model, will use fallback")
        else:
            logger.info("YOLO-E v8l model loaded successfully")
    except Exception as e:
        logger.error(f"Error during YOLO-E v8l model loading: {e}")
        logger.warning("Will use fallback models for inference")
    
    yield
    
    # Cleanup on shutdown
    global video_cap
    if video_cap is not None:
        video_cap.release()
        logger.info("Video capture released")


app = FastAPI(lifespan=lifespan)

# CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and video capture
video_cap = None
yolo_e_model = None


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


def load_yolo_e_model():
    """Load YOLO-E v8l model for efficient instance segmentation with text prompts"""
    global yolo_e_model
    try:
        logger.info("Loading YOLO-E v8l model...")

        # Check if model file already exists
        model_path = "yoloe-v8l-seg.pt"

        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"YOLO-E v8l model already exists: {model_path} "
                       f"({file_size:.1f} MB)")
        else:
            # Download fresh model
            logger.info("YOLO-E v8l model not found, downloading...")
            if not download_yolo_e_v8l_model_direct():
                logger.error("Failed to download YOLO-E v8l model")
                return False

        # Load the model
        try:
            yolo_e_model = YOLOE(model_path)

            # Set default classes to limited retail items
            try:
                # Try to get text embeddings safely
                try:
                    text_pe = yolo_e_model.get_text_pe(YOLOE_CLASSES)
                    yolo_e_model.set_classes(YOLOE_CLASSES, text_pe)
                    logger.info(f"YOLO-E v8l model loaded with limited classes: "
                               f"{YOLOE_CLASSES}")
                except Exception as pe_error:
                    logger.warning(f"Could not get text embeddings: {pe_error}")
                    # Fallback: try setting classes without text embeddings
                    try:
                        yolo_e_model.set_classes(YOLOE_CLASSES)
                        logger.info(f"YOLO-E v8l model loaded with limited classes (fallback): "
                                   f"{YOLOE_CLASSES}")
                    except Exception as fallback_error:
                        logger.warning(f"Could not set default classes: {fallback_error}")
                        logger.info("YOLO-E v8l model loaded successfully")
            except Exception as class_error:
                logger.warning(f"Could not set default classes: {class_error}")
                logger.info("YOLO-E v8l model loaded successfully")

            return True
        except Exception as load_error:
            logger.error(f"Failed to load YOLO-E v8l model: {load_error}")
            return False

    except Exception as e:
        logger.error(f"Failed to load YOLO-E v8l model: {e}")
        return False


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert an OpenCV frame (numpy array) to a base64 encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def load_video(video_path: str):
    """Load the video file into a global OpenCV VideoCapture object."""
    global video_cap
    try:
        logger.info(f"Loading video: {video_path}")

        # Check if video file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False

        # Get file size for progress indication
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size / (1024*1024):.2f} MB")

        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            logger.error("Failed to open video file")
            return False

        # Get video properties
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video loaded successfully: {total_frames} frames, "
                   f"{fps:.2f} FPS, {duration:.2f}s duration")
        return True
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        return False


def get_frame_at_time(target_time: float) -> np.ndarray:
    """Get frame at specific time in video"""
    global video_cap
    if video_cap is None:
        return None

    fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    target_frame = int(target_time * fps)

    if target_frame >= total_frames:  # Loop video if end is reached
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_frame = 0
        logger.info("Video looped to beginning.")

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = video_cap.read()

    if ret:
        return frame
    return None


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

        # Convert frame to PIL Image for YOLOE
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Parse text prompt into classes and set them on the model
        if text_prompt:
            try:
                prompt_classes = [cls.strip() for cls in text_prompt.split(',')
                                if cls.strip()]
                if prompt_classes:
                    # Try to get text embeddings safely
                    try:
                        text_pe = yolo_e_model.get_text_pe(prompt_classes)
                        yolo_e_model.set_classes(prompt_classes, text_pe)
                        logger.info(f"YOLO-E text prompts set: {prompt_classes}")
                    except Exception as pe_error:
                        logger.warning(f"Could not get text embeddings: {pe_error}")
                        # Fallback: try setting classes without text embeddings
                        try:
                            yolo_e_model.set_classes(prompt_classes)
                            logger.info(f"YOLO-E text prompts set (fallback): {prompt_classes}")
                        except Exception as fallback_error:
                            logger.warning(f"Could not set text prompts at all: {fallback_error}")
                else:
                    logger.warning("No valid classes found in text prompt")
            except Exception as e:
                logger.warning(f"Could not set text prompts: {e}")

        # Run YOLO-E inference - exactly like your reference code
        results = yolo_e_model.predict(pil_image, conf=0.1, verbose=False)

        if not results or len(results) == 0:
            return {
                "detections": [],
                "total_objects": 0,
                "annotated_frame": frame.copy(),
                "clicked_object": None,
                "text_prompt_used": text_prompt if text_prompt else "default_classes"
            }

        # Process results from YOLOE
        result = results[0]
        detections = []
        annotated_frame = frame.copy()

        # Check if we have detections
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for i, box in enumerate(boxes):
                try:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
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

                    # Extract segmentation mask if available
                    mask = None
                    if masks is not None and i < len(masks):
                        try:
                            if hasattr(masks, 'xy'):
                                mask = masks.xy[i].tolist()
                            elif hasattr(masks, 'data'):
                                # Convert mask data to polygon format
                                mask_data = masks.data[i].cpu().numpy()
                                # Find contours in the mask
                                contours, _ = cv2.findContours(
                                    mask_data.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                )
                                if contours:
                                    # Use the largest contour
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    mask = largest_contour.squeeze().tolist()
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

        # Set classes and text prompts using the clean approach from reference code
        try:
            # Try to get text embeddings safely
            try:
                text_pe = yolo_e_model.get_text_pe(prompt_classes)
                yolo_e_model.set_classes(prompt_classes, text_pe)
                logger.info(f"YOLO-E v8l text prompts set: {prompt_classes}")
            except Exception as pe_error:
                logger.warning(f"Could not get text embeddings: {pe_error}")
                # Fallback: try setting classes without text embeddings
                try:
                    yolo_e_model.set_classes(prompt_classes)
                    logger.info(f"YOLO-E v8l text prompts set (fallback): {prompt_classes}")
                except Exception as fallback_error:
                    logger.warning(f"Could not set text prompts, using default: {fallback_error}")
        except Exception as e:
            logger.warning(f"Could not set text prompts, using default: {e}")

        # Run YOLO-E inference - exactly like your reference code
        results = yolo_e_model.predict(pil_image, conf=confidence, verbose=False)

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




@app.get("/")
async def read_root():
    return {"message": "Retail Vision Backend API"}


@app.get("/api/video-status", response_model=VideoStatus)
async def get_video_status():
    """Get the current status of the loaded video."""
    if video_cap is None:
        return VideoStatus(is_loaded=False, total_frames=0, fps=0, duration=0)

    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
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
        # Get frame at the specified video time
        frame = get_frame_at_time(request.video_time)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        # Log coordinate information for debugging
        logger.info(f"Click coordinates: ({request.x}, {request.y}) on frame "
                   f"{frame.shape[1]}x{frame.shape[0]}")
        logger.info(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
        logger.info(f"Requested frame dimensions: "
                   f"{request.frame_width}x{request.frame_height}")

        # Run YOLO-E inference - simple and direct like your reference code
        inference_result = run_yolo_e_inference(
            frame, request.x, request.y, request.text_prompt
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
            video_time=request.video_time,
            clicked_pixel={"x": request.x, "y": request.y},
            detections=inference_result.get("detections", []),
            frame_base64=frame_base64,
            annotated_frame_base64=annotated_frame_base64,
            clicked_object=inference_result.get("clicked_object"),
            inference_type="YOLO-E",
            text_prompt_used=inference_result.get("text_prompt_used")
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

        # Update the model's text prompts using the correct YOLOE pattern
        try:
            # Try to get text embeddings safely
            try:
                text_pe = yolo_e_model.get_text_pe(prompt_classes)
                yolo_e_model.set_classes(prompt_classes, text_pe)
                logger.info(f"YOLO-E text prompts updated: {prompt_classes}")
            except Exception as pe_error:
                logger.warning(f"Could not get text embeddings: {pe_error}")
                # Fallback: try setting classes without text embeddings
                try:
                    yolo_e_model.set_classes(prompt_classes)
                    logger.info(f"YOLO-E text prompts updated (fallback): {prompt_classes}")
                except Exception as fallback_error:
                    logger.warning(f"Could not update text prompts: {fallback_error}")
                    raise HTTPException(
                        status_code=500, detail=f"Could not update text prompts: {fallback_error}"
                    )
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
        if video_cap is not None:
            # Get current frame position
            current_frame = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame <= 0:
                # If at beginning, get first frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = video_cap.read()
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

        # Run YOLO-E v8l inference
        inference_result = run_yolo_e_v8l_inference(
            frame, request.text_prompt, request.confidence
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
