# YOLO-E v8l Endpoint Documentation

This document describes the new YOLO-E v8l endpoint that has been added to your FastAPI backend in `main.py`.

## Overview

The YOLO-E v8l endpoint provides efficient instance segmentation with text prompts using the YOLO-E v8l model. This endpoint follows the same clean, working approach as your reference code in `yoloe_simple_clean.py`.

## New Endpoints Added

### 1. YOLO-E v8l Inference Endpoint

**Endpoint:** `POST /api/inference/yolo-e-v8l`

**Purpose:** Run YOLO-E v8l inference on video frames with custom text prompts

**Request Body:**
```json
{
    "text_prompt": "cap, t shirt, backpack",
    "confidence": 0.1
}
```

**Parameters:**
- `text_prompt` (string): Comma-separated list of object classes to detect
- `confidence` (float, optional): Confidence threshold for detection (default: 0.1)

**Response:**
```json
{
    "timestamp": 1234567890.123,
    "video_time": 0.0,
    "clicked_pixel": {"x": 0, "y": 0},
    "detections": [...],
    "frame_base64": "base64_encoded_frame",
    "annotated_frame_base64": "base64_encoded_annotated_frame",
    "clicked_object": null,
    "inference_type": "YOLO-E-v8l",
    "text_prompt_used": "cap, t shirt, backpack"
}
```

### 2. YOLO-E v8l Status Endpoint

**Endpoint:** `GET /api/yolo-e-v8l/status`

**Purpose:** Check the status and availability of the YOLO-E v8l model

**Response:**
```json
{
    "model_loaded": true,
    "model_name": "YOLO-E v8l",
    "status": "Loaded and ready",
    "timestamp": 1234567890.123,
    "available_classes": ["cap", "t shirt", "backpack"]
}
```

## How It Works

### 1. Model Loading
- The YOLO-E v8l model (`yoloe-v8l-seg.pt`) is automatically loaded during application startup
- If the model file doesn't exist, it will be automatically downloaded from the official repository
- The model is loaded into the global variable `yolo_e_v8l_model`

### 2. Text Prompt Processing
- Text prompts are parsed into individual class names
- The model's classes are set using `model.set_classes(NAMES, model.get_text_pe(NAMES))`
- This follows the exact pattern from your working reference code

### 3. Inference Process
- Frames are converted from OpenCV (BGR) to PIL (RGB) format for YOLO-E
- Inference is run with the specified confidence threshold
- Results are converted to supervision format for better mask handling
- Masks and labels are applied using supervision's annotators
- Results are converted back to OpenCV format for consistency

### 4. Fallback Handling
- If supervision annotation fails, the system falls back to basic OpenCV annotation
- This ensures robustness even if the advanced annotation features aren't available

## Integration with Existing Code

The new endpoint integrates seamlessly with your existing FastAPI backend:

- **Global Variables:** Added `yolo_e_v8l_model` alongside existing models
- **Startup Function:** Automatically loads the v8l model during application startup
- **Error Handling:** Consistent with your existing error handling patterns
- **Logging:** Uses the same logging configuration for consistency
- **Response Format:** Compatible with your existing `DetectionResult` model

## Usage Examples

### Python Client
```python
import requests

# Test the endpoint
response = requests.post(
    "http://localhost:8000/api/inference/yolo-e-v8l",
    json={
        "text_prompt": "cap, t shirt, backpack",
        "confidence": 0.1
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Detected {result['total_objects']} objects")
    print(f"Using text prompt: {result['text_prompt_used']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/inference/yolo-e-v8l" \
     -H "Content-Type: application/json" \
     -d '{"text_prompt": "cap, t shirt, backpack", "confidence": 0.1}'
```

### Check Model Status
```bash
curl "http://localhost:8000/api/yolo-e-v8l/status"
```

## Testing

A test script `test_yoloe_v8l_endpoint.py` has been created to verify the endpoint functionality:

```bash
python test_yoloe_v8l_endpoint.py
```

This script will:
1. Check the model status
2. Test the inference endpoint with sample text prompts
3. Display the results and any detected objects

## Model Files

The endpoint automatically manages the required model files:

- **Primary Model:** `yoloe-v8l-seg.pt` (YOLO-E v8l segmentation model)
- **MobileCLIP Integration:** Automatically uses existing MobileCLIP files for text prompts
- **Automatic Download:** Downloads missing models from official repositories

## Performance Features

- **Efficient Processing:** Uses the lightweight v8l model for fast inference
- **Text Prompt Optimization:** Leverages MobileCLIP for efficient text-based object detection
- **Mask Generation:** Produces high-quality segmentation masks
- **Confidence Control:** Configurable confidence thresholds for detection sensitivity

## Error Handling

The endpoint includes comprehensive error handling:

- **Model Loading Errors:** Graceful fallback if model fails to load
- **Inference Errors:** Detailed error messages for debugging
- **Text Prompt Errors:** Validation of input text prompts
- **Fallback Mechanisms:** Automatic fallback to basic inference if advanced features fail

## Dependencies

The new functionality requires:

- `ultralytics` (for YOLO-E model)
- `supervision` (for advanced annotation)
- `PIL` (for image processing)
- `numpy` and `cv2` (for frame handling)

All dependencies are already included in your existing setup.

## Next Steps

1. **Start the backend server** to load the new model
2. **Test the endpoint** using the provided test script
3. **Integrate with your frontend** to use the new YOLO-E v8l capabilities
4. **Customize text prompts** for your specific use cases

The endpoint is now ready to provide efficient, text-prompted object detection and segmentation using the YOLO-E v8l model! 🚀







