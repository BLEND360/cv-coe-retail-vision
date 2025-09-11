# Multi-stage Dockerfile for Retail Vision Application

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files
COPY retail-vision-ui/package*.json ./

# Install dependencies
RUN npm install --production

# Copy source code
COPY retail-vision-ui/src ./src
COPY retail-vision-ui/public ./public
COPY retail-vision-ui/tsconfig.json ./

# Build the application
RUN npm run build

# Stage 2: Python backend with system dependencies
FROM python:3.11-slim AS backend-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including OpenGL for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY retail-vision-ui/backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Stage 3: Production image
FROM backend-base AS production

# Copy backend application
COPY retail-vision-ui/backend/ ./backend/

# Copy built frontend from frontend-build stage
COPY --from=frontend-build /app/frontend/build /var/www/html

# Copy video files if they exist (will fail silently if directory doesn't exist)
COPY static/videos/ ./backend/static/videos/

# Create necessary directories
RUN mkdir -p backend/static/videos backend/models

# Set proper permissions
RUN chmod +x backend/main.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the backend service
CMD ["python", "backend/main.py"]