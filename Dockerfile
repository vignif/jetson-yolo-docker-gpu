# NVIDIA L4T PyTorch image for Jetson Nano (R32 = JetPack 4.x with CUDA 10.2)
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Set CUDA library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /app

# Remove Kitware APT repository and install system dependencies
RUN sed -i '/kitware/d' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true && \
    rm -f /etc/apt/sources.list.d/kitware*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
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
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    find /usr -name "libmpi_cxx.so*" 2>/dev/null && \
    ln -sf /usr/lib/aarch64-linux-gnu/openmpi/lib/libmpi_cxx.so.1 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20 || \
    ln -sf $(find /usr -name "libmpi_cxx.so.1*" 2>/dev/null | head -1) /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20

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
