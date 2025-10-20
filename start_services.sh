#!/bin/bash

# Startup script for Docker container
# Runs both backend and frontend services

set -e

echo "Starting Retail Vision Services"
echo "==============================="

# Start backend
echo "Starting backend..."
cd /app/retail-vision-ui/backend
nohup python run_backend.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Start frontend
echo "Starting frontend..."
cd /app/retail-vision-ui
nohup npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "Retail Vision is running!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

# Function to handle shutdown
cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
