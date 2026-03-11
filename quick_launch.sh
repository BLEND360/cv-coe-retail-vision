#!/bin/bash

# Quick Launch Script for Retail Vision
# Usage:
#   ./quick_launch.sh                    # Launch with Under Armour (default)
#   ./quick_launch.sh under-armour       # Launch with Under Armour
#   ./quick_launch.sh blend360           # Launch with BLEND360

set -e

BRAND="${1:-blend360}"

# Map brand to video file
case "$BRAND" in
  under-armour)
    VIDEO_PATH="../public/Under-Armour.mp4"
    echo "Brand: Under Armour"
    ;;
  blend360)
    VIDEO_PATH="../public/The BLEND360 Approach.mp4"
    echo "Brand: BLEND360"
    ;;
  *)
    echo "Unknown brand: $BRAND"
    echo "Available brands: under-armour, blend360"
    exit 1
    ;;
esac

echo "Quick Launch - Retail Vision"
echo "============================"

# Kill processes on ports 3000 and 8000
echo "Clearing ports..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
echo "Ports cleared"

# Start backend
echo "Starting backend..."
cd retail-vision-ui/backend
source venv/bin/activate
VIDEO_PATH="$VIDEO_PATH" nohup python run_backend.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ../..
echo "Backend started (PID: $BACKEND_PID)"

# Start frontend
echo "Starting frontend..."
cd retail-vision-ui
REACT_APP_BRAND="$BRAND" nohup npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "Retail Vision is running!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    echo "Services stopped"
}

trap cleanup EXIT INT TERM

# Wait
while true; do
    sleep 1
done
