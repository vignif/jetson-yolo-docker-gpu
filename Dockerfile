# NVIDIA L4T PyTorch image for Jetson (includes CUDA/TensorRT)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# System deps for GStreamer + camera + OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create models directory
RUN mkdir -p /app/models

# Run GPU health check during build to validate setup
RUN echo "Running GPU health check during build..." && \
    python3 /app/gpu_health_check.py || echo "GPU check failed (expected if building on non-GPU host)"

EXPOSE 8000

# Use entrypoint script that validates GPU on container start
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
