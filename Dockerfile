# Multi-stage Dockerfile for Retail Vision Application
# Supports both CPU and GPU (NVIDIA CUDA) automatically
#
# CPU build (default):  docker build -t retail-vision .
# GPU build:            docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 -t retail-vision-gpu .
#
# CPU run:              docker run -p 8000:8000 retail-vision
# GPU run:              docker run --gpus all -p 8000:8000 retail-vision-gpu

ARG BASE_IMAGE=python:3.11-slim

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files
COPY retail-vision-ui/package*.json ./

# Install ALL dependencies (devDependencies needed for react-scripts build)
RUN npm ci

# Copy source code
COPY retail-vision-ui/src ./src
COPY retail-vision-ui/public ./public
COPY retail-vision-ui/tsconfig.json ./

# Build the application — empty API URL makes frontend use relative URLs (same origin)
ENV REACT_APP_API_URL=""
RUN npm run build

# Stage 2: Python backend with system dependencies
FROM ${BASE_IMAGE} AS backend-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including OpenGL for OpenCV
# Works on both debian-based (python:slim) and ubuntu-based (nvidia/cuda) images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3-dev \
    python3-pip \
    libgl1-mesa-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python/pip are available (nvidia images use python3)
RUN if ! command -v python &> /dev/null; then \
        ln -s /usr/bin/python3 /usr/bin/python && \
        ln -s /usr/bin/pip3 /usr/bin/pip; \
    fi

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
# For GPU images, PyTorch with CUDA is installed via requirements
COPY retail-vision-ui/backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Stage 3: Production image
FROM backend-base AS production

# Copy backend application
COPY retail-vision-ui/backend/ ./backend/

# Copy built frontend from frontend-build stage
COPY --from=frontend-build /app/frontend/build /var/www/html

# Copy video file for the demo
COPY retail-vision-ui/public/Under-Armour.mp4 ./public/Under-Armour.mp4

# Copy model files (mobileclip for text embeddings)
COPY retail-vision-ui/backend/mobileclip_blt.pt ./backend/mobileclip_blt.pt

# Create necessary directories
RUN mkdir -p backend/static/videos backend/models

# Set environment variables for container deployment
ENV VIDEO_PATH=/app/public/Under-Armour.mp4 \
    FRONTEND_DIR=/var/www/html

# Set working directory to backend for model path resolution
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start with multiple workers for better concurrency
# Workers handle parallel requests while inference runs in thread pools
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
