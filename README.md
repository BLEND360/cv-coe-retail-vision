# Retail Vision 🛍️

A cutting-edge computer vision application that enables users to click on objects in videos and purchase them immediately. Built with state-of-the-art object segmentation models and a modern web interface.

## 🎯 Project Overview

Retail Vision transforms passive video watching into an interactive shopping experience. Using advanced computer vision and machine learning techniques, the application:

- **Detects and segments objects** in real-time video streams
- **Enables click-to-purchase** functionality for identified products
- **Provides instant product recognition** using YOLO-E segmentation models
- **Offers seamless integration** between video content and e-commerce

## 🏗️ Architecture

The project consists of three main components:

### 1. **Backend API** (FastAPI + Python)
- **Object Detection & Segmentation**: Powered by YOLO-E (You Only Look Once Edge) models
- **Video Processing**: Real-time frame analysis and object tracking
- **RESTful API**: FastAPI endpoints for segmentation data
- **Model Management**: Support for multiple YOLO-E model variants

### 2. **Frontend UI** (React + TypeScript)
- **Video Player**: Interactive video interface with clickable objects
- **Segmentation Visualization**: Real-time display of detected objects
- **Shopping Integration**: Seamless product selection and purchase flow
- **Responsive Design**: Modern Material-UI components

### 3. **Computer Vision Models**
- **YOLO-E v8l-seg**: Efficient edge-optimized instance segmentation model
- **MobileCLIP**: Text-to-image understanding for enhanced object recognition

## 🚀 Features

- **Real-time Object Detection**: Instant recognition of products in video content
- **Interactive Segmentation**: Clickable object boundaries with confidence scores
- **Multi-model Support**: Choose between speed and accuracy based on use case
- **Video Timeline Integration**: Navigate to specific timestamps for object analysis
- **Cross-platform Compatibility**: Works on desktop and mobile devices
- **Modern UI/UX**: Intuitive interface with Material Design principles
- **Docker Support**: Easy deployment with containerization

## 📋 Prerequisites

Before running this project, ensure you have:

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Git** for version control
- **Docker** (optional, for easy deployment)
- **FFmpeg** for video processing (optional, for additional video formats)

## 🛠️ Installation

### Option 1: Local Development Setup

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd cv-coe-retail-vision
```

#### 2. Backend Setup

##### Automated Installation (Recommended)
```bash
cd retail-vision-ui/backend

# Run the installation script
./install_dependencies.sh

# Or use Python version
python install_yoloe_dependencies.py
```

##### Manual Installation
```bash
cd retail-vision-ui/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install YOLO-E specific dependencies
pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/CLIP"
pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip"
pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/lvis-api"
pip install "git+https://github.com/THU-MIG/yoloe.git"

# Download MobileCLIP model (if not present)
curl -L -o mobileclip_blt.pt https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
```

##### Verification Steps
```bash
# Test imports
python -c "
import supervision as sv
from ultralytics import YOLOE
import torch
from PIL import Image
print('✅ All imports successful!')
"

# Check model files
ls -lh *.pt *.ts
# Should show:
# - yoloe-v8l-seg.pt (~107MB)
# - mobileclip_blt.pt (~599MB)  
# - mobileclip_blt.ts (~380MB)
```

#### 3. Frontend Setup
```bash
cd retail-vision-ui

# Install dependencies
npm install
```

### Option 2: Docker Deployment

#### Quick Start
```bash
# Build and run everything
docker-compose up -d

# Access the app
# Frontend: http://localhost:3000
# Backend: http://localhost:8000

# Stop everything
docker-compose down
```

#### Backend Only
```bash
# Build and run backend
docker build -t retail-vision .
docker run -p 8000:8000 retail-vision
```

## 🚀 Running the Application

### Local Development

#### 1. Start the Backend Server
```bash
cd retail-vision-ui/backend
source venv/bin/activate  # or activate on Windows

# Option 1: Using the runner script
python run_backend.py

# Option 2: Direct uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

#### 2. Start the Frontend Development Server
```bash
cd retail-vision-ui
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Docker Deployment
```bash
# Start everything
docker-compose up -d

# Access: Frontend http://localhost:3000, Backend http://localhost:8000
# Stop: docker-compose down
# API Docs: http://localhost/docs
```

## 🌐 Access Points

Once running, the application is available at:

- **Frontend Application**: 
  - Local: http://localhost:3000
  - Docker: http://localhost (port 80)
- **Backend API**: 
  - Local: http://localhost:8000
  - Docker: http://localhost:8000
- **Interactive API Documentation**: 
  - Local: http://localhost:8000/docs
  - Docker: http://localhost/docs
- **Health Check**: 
  - Local: http://localhost:8000/
  - Docker: http://localhost/health

## 🔧 Configuration

### Model Selection
The application uses YOLO-E models for efficient edge-optimized inference:

1. **Primary Model**:
   - `yoloe-v8l-seg.pt` (~107MB): Efficient edge-optimized segmentation model
   - Located in: `retail-vision-ui/backend/yoloe-v8l-seg.pt`
   - Automatically downloaded on first startup if missing

2. **Supporting Models**:
   - `mobileclip_blt.pt` (~599MB): MobileCLIP PyTorch model for text encoding
   - `mobileclip_blt.ts` (~380MB): MobileCLIP TorchScript model for text processing
   - Both required for text prompt functionality

### Video Configuration
Update the video path in `retail-vision-ui/backend/main.py`:
```python
video_path = "static/videos/The BLEND360 Approach.mp4"
```

### Environment Variables
Create `.env` files for configuration:

**Backend (.env)**:
```bash
VIDEO_PATH=/path/to/video.mp4
MODEL_PATH=/path/to/model.pt
ENVIRONMENT=development
PYTHONUNBUFFERED=1
```

**Frontend (.env.local)**:
```bash
REACT_APP_API_URL=http://localhost:8000
```

### YOLO-E Model Details
The application uses YOLO-E v8l with the following features:
- **Text Prompts**: Supports custom object classes via text descriptions
- **Default Classes**: `["laptop", "headphones", "glasses", "blazer", "desk", "watch", "monitor", "trash can", "chair", "shirt"]`
- **Confidence Threshold**: 0.1 (configurable)
- **MobileCLIP Integration**: Enables text-to-image understanding for enhanced object recognition

## 📊 API Endpoints

### Core Endpoints

- **`GET /`** - Health check
- **`GET /api/status`** - Video and model status
- **`POST /api/inference`** - Get inference for specific video time
- **`GET /api/inference/latest`** - Get latest inference result

### Example API Usage

**Get inference for timestamp:**
```bash
curl -X POST "http://localhost:8000/api/inference" \
     -H "Content-Type: application/json" \
     -d '{"timestamp": 10.5}'
```

**Response:**
```json
{
  "timestamp": 10.5,
  "detections": [
    {
      "class_id": 73,
      "class_name": "laptop",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300],
      "mask": "base64_encoded_mask"
    }
  ],
  "frame_base64": "base64_encoded_frame"
}
```

## 🛠️ Development Scripts

### Frontend Scripts
| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm start` | Start development server (alias) |
| `npm run build` | Build production bundle |
| `npm run test` | Run test suite |
| `npm run lint` | Check code quality with ESLint |
| `npm run type-check` | Run TypeScript type checking |
| `npm run clean` | Remove build artifacts |
| `npm run preview` | Build and serve production build locally |

### Backend Scripts
| Command | Description |
|---------|-------------|
| `python run_backend.py` | Start backend with auto-reload |
| `uvicorn main:app --reload` | Manual backend startup |
| `python main.py` | Direct Python execution |

## 📁 Project Structure

```
cv-coe-retail-vision/
├── retail-vision-ui/           # Main application directory
│   ├── backend/                # FastAPI backend
│   │   ├── main.py            # Main API server
│   │   ├── run_backend.py     # Backend startup script
│   │   ├── requirements.txt   # Python dependencies
│   │   ├── static/            # Static files (videos, models)
│   │   │   └── videos/        # Video files
│   │   └── venv/              # Python virtual environment
│   ├── src/                   # React frontend source code
│   │   ├── components/        # React components
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── InferencePanel.tsx
│   │   │   └── ShoppingCart.tsx
│   │   ├── App.tsx            # Main application component
│   │   └── index.tsx          # Application entry point
│   ├── public/                # Static files
│   ├── package.json           # Node.js dependencies and scripts
│   └── tsconfig.json          # TypeScript configuration
├── models/                     # YOLO model files
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Single container Dockerfile
├── nginx.conf                  # Nginx configuration
├── supervisord.conf            # Process management configuration
├── requirements.txt            # Root Python dependencies
├── install_dependencies.sh     # Dependency installation script
└── venv/                      # Root Python virtual environment
```


## 🐛 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Kill processes on ports 3000 and 8000
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# Or use different ports
npm run dev -- --port 3001
uvicorn main:app --port 8001
```

#### 2. Python Dependencies Issues
```bash
cd retail-vision-ui/backend
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Node Modules Issues
```bash
cd retail-vision-ui
rm -rf node_modules package-lock.json
npm install
```

#### 4. Docker Issues
```bash
# Clean up and rebuild
docker-compose down -v
docker-compose up --build
```

#### 5. Model Loading Issues
- Ensure YOLO model files exist in the backend directory
- Check file permissions
- Verify CUDA installation if using GPU
- Use smaller models for limited resources


### Health Checks
```bash
# Check if services are running
curl http://localhost:8000/  # Backend
curl http://localhost:3000/  # Frontend
```

### Performance Optimization

- **GPU Acceleration**: Install CUDA for faster inference
- **Model Selection**: Use smaller models for real-time performance
- **Memory Management**: Monitor RAM usage with larger models
- **Caching**: Implement result caching for repeated requests

## 🔒 Security Considerations

- **Input Validation**: All API inputs are validated
- **CORS Configuration**: Configured for development; restrict in production
- **Model Security**: Use trusted YOLO models from official sources
- **Rate Limiting**: Implement API rate limiting for production use
- **Authentication**: Add authentication for production deployment

## 🧪 Testing

### Backend Testing
```bash
cd retail-vision-ui/backend
python -m pytest  # If pytest is configured
```

### Frontend Testing
```bash
cd retail-vision-ui
npm test
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/

# Test inference endpoint
curl -X POST "http://localhost:8000/api/inference" \
     -H "Content-Type: application/json" \
     -d '{"timestamp": 0}'
```

## 🚧 Known Issues & Limitations

- **Model Loading**: Large models may take time to load initially
- **Video Format Support**: Limited to common video formats (MP4, AVI)
- **Real-time Processing**: Processing speed depends on hardware capabilities
- **Object Classes**: Currently focused on general object detection
- **Memory Usage**: YOLO-E models require significant RAM

## 📈 Performance Tips

- **Hardware**: Use GPU acceleration for faster inference
- **Model Size**: Choose appropriate model size based on performance needs
- **Batch Processing**: Implement batch inference for multiple frames
- **Caching**: Add result caching for repeated timestamp requests
- **Optimization**: Use model quantization for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add tests for new features
- Update documentation for API changes
- Ensure Docker builds work correctly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLO models and ultralytics library
- **THU-MIG**: For the YOLO-E implementation
- **FastAPI**: For the modern, fast web framework
- **React**: For the frontend framework
- **Material-UI**: For the beautiful UI components
- **Apple**: For the MobileCLIP model

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions
- Review the troubleshooting section above

---

**Retail Vision** - Transforming video watching into interactive shopping experiences! 🎥🛒

*Built with ❤️ using YOLO-E, FastAPI, React, and Docker*