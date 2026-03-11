#!/bin/bash
# Setup script for Retail Vision
# Run this after cloning the repo to get everything ready

set -e

echo "Retail Vision - Setup"
echo "====================="

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_CMD="python3"
    echo "Using Python $PY_VERSION (Python 3.11+ recommended)"
else
    echo "Error: Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js not found. Please install Node.js 18+"
    exit 1
fi

echo ""
echo "1/4 Setting up backend..."
cd retail-vision-ui/backend
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "  Created virtual environment"
fi
source venv/bin/activate
pip install --quiet -r requirements.txt
echo "  Backend dependencies installed"

# Download models if not present
if [ ! -f "mobileclip_blt.pt" ]; then
    echo "  Downloading mobileclip_blt.pt (571MB)..."
    curl -L -o mobileclip_blt.pt \
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
    echo "  mobileclip_blt.pt downloaded"
else
    echo "  mobileclip_blt.pt already exists"
fi

if [ ! -f "yoloe-v8l-seg.pt" ]; then
    echo "  Downloading yoloe-v8l-seg.pt (107MB)..."
    curl -L -o yoloe-v8l-seg.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt"
    echo "  yoloe-v8l-seg.pt downloaded"
else
    echo "  yoloe-v8l-seg.pt already exists"
fi

cd ../..

echo ""
echo "2/4 Setting up frontend..."
cd retail-vision-ui
npm ci --quiet
echo "  Frontend dependencies installed"
cd ..

echo ""
echo "3/4 Verifying files..."
if [ -f "retail-vision-ui/public/Under-Armour.mp4" ]; then
    echo "  Video file: OK"
else
    echo "  WARNING: retail-vision-ui/public/Under-Armour.mp4 not found"
fi

if [ -f "retail-vision-ui/backend/mobileclip_blt.pt" ]; then
    echo "  MobileCLIP model: OK"
else
    echo "  WARNING: mobileclip_blt.pt not found (text prompts will not work)"
fi

if [ -f "retail-vision-ui/backend/yoloe-v8l-seg.pt" ]; then
    echo "  YOLO-E v8l model: OK"
else
    echo "  WARNING: yoloe-v8l-seg.pt not found (will auto-download on first run)"
fi

echo ""
echo "4/4 Setup complete!"
echo ""
echo "To run locally:"
echo "  ./quick_launch.sh"
echo ""
echo "To build Docker container:"
echo "  docker build -t retail-vision ."
echo "  docker run -p 8000:8000 retail-vision"
