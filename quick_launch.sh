#!/bin/bash

# Quick Launch Script for Retail Vision (Under Armour)

set -e

echo "Quick Launch - Retail Vision (Under Armour)"
echo "============================================"

# Kill processes on ports 3000 and 8000
echo "Clearing ports..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
echo "Ports cleared"

# Start backend
echo "Starting backend..."
cd retail-vision-ui/backend
source venv/bin/activate
VIDEO_PATH="../public/Under-Armour.mp4" nohup python run_backend.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ../..
echo "Backend started (PID: $BACKEND_PID)"

# Start frontend
echo "Starting frontend..."
cd retail-vision-ui
nohup npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "Retail Vision (Under Armour) is running!"
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
