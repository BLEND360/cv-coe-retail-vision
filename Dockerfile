# Multi-stage Dockerfile for Retail Vision Application
# Supports both CPU and GPU (NVIDIA CUDA) automatically
#
# CPU build (default):   docker build -t retail-vision .
# GPU build:             docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 -t retail-vision-gpu .
#
# CPU run:               docker run -p 8000:8000 retail-vision
# GPU run:               docker run --gpus all -p 8000:8000 retail-vision-gpu

ARG BASE_IMAGE=python:3.11-slim

# Stage 0: Download YOLOE source (runs on native arch, no QEMU)
FROM --platform=$BUILDPLATFORM alpine:3.19 AS yoloe-src
RUN apk add --no-cache curl
ARG YOLOE_COMMIT=40cd606cabdbe2b566d6f14a6b162c89206e9a1b
RUN curl --fail -L -o /tmp/yoloe.tar.gz \
        "https://github.com/THU-MIG/yoloe/archive/${YOLOE_COMMIT}.tar.gz" && \
    mkdir /tmp/yoloe && \
    tar xz -C /tmp/yoloe --strip-components=1 -f /tmp/yoloe.tar.gz && \
    rm /tmp/yoloe.tar.gz

# Stage 1: Build React frontend
FROM --platform=$BUILDPLATFORM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files
COPY retail-vision-ui/package*.json ./

# Install ALL dependencies (devDependencies needed for react-scripts build)
RUN npm ci

# Copy source code
COPY retail-vision-ui/src ./src
COPY retail-vision-ui/public ./public
COPY retail-vision-ui/tsconfig.json ./

# Build the application
ENV REACT_APP_API_URL="" \
    REACT_APP_BRAND=blend360
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
RUN command -v python > /dev/null 2>&1 || ln -sf /usr/bin/python3 /usr/bin/python; \
    command -v pip > /dev/null 2>&1 || ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
# For GPU images, PyTorch with CUDA is installed via requirements
COPY retail-vision-ui/backend/requirements.txt ./backend/

# Copy YOLOE source from native download stage (avoids QEMU network issues)
COPY --from=yoloe-src /tmp/yoloe /tmp/yoloe
RUN pip install --no-cache-dir \
        /tmp/yoloe/third_party/CLIP \
        /tmp/yoloe/third_party/ml-mobileclip \
        /tmp/yoloe/third_party/lvis-api \
        /tmp/yoloe && \
    rm -rf /tmp/yoloe

# Install remaining (non-git) requirements
RUN grep -v 'git+' backend/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Stage 3: Production image
FROM backend-base AS production

# Copy backend application
COPY retail-vision-ui/backend/ ./backend/

# Copy built frontend from frontend-build stage
COPY --from=frontend-build /app/frontend/build /var/www/html

# Copy video files for the demo
COPY "retail-vision-ui/public/The BLEND360 Approach.mp4" ./public/

# Download model files (gitignored, so they must be fetched during build)
# --fail makes curl return exit code 22 on HTTP errors (e.g. 404, 403)
# so the build fails fast instead of saving an error page as the model file.
RUN curl --fail -L -o backend/yoloe-v8l-seg.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt" && \
    curl --fail -L -o backend/mobileclip_blt.pt \
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt" && \
    echo "Model file sizes:" && \
    ls -lh backend/yoloe-v8l-seg.pt backend/mobileclip_blt.pt && \
    test $(stat -c%s backend/yoloe-v8l-seg.pt 2>/dev/null || stat -f%z backend/yoloe-v8l-seg.pt) -gt 50000000 && \
    test $(stat -c%s backend/mobileclip_blt.pt 2>/dev/null || stat -f%z backend/mobileclip_blt.pt) -gt 50000000

# Create necessary directories
RUN mkdir -p backend/models

# Set environment variables for container deployment
ENV FRONTEND_DIR=/var/www/html \
    VIDEO_PATH="/app/public/The BLEND360 Approach.mp4"

# Set working directory to backend for model path resolution
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
