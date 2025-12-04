# NVIDIA L4T PyTorch image for Jetson (includes CUDA/TensorRT)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# System deps for GStreamer + camera + OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
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

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
