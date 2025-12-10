# Jetson Camera Streaming

Real-time camera streaming application for NVIDIA Jetson with Raspberry Pi Camera Module v2.

## Features

- 🎥 Hardware-accelerated video capture via GStreamer with NVMM
- 🌐 WebSocket-based streaming for low latency
- 👥 Multiple simultaneous viewers support
- 📊 Health check and statistics endpoints
- 🐳 Fully containerized with Docker

## Architecture

The application is structured into modular components:

- **`camera.py`** - Camera capture with GStreamer pipeline management
- **`encoder.py`** - JPEG frame encoding with configurable quality
- **`client_manager.py`** - WebSocket client lifecycle and broadcasting
- **`streaming.py`** - Main streaming service orchestration
- **`main.py`** - FastAPI application and HTTP/WebSocket endpoints

## Quick Start

### Deploy to Jetson

```bash
./deploy.sh jetson 192.168.1.67 jetson-webcam
```

### Access the Stream

Open in your browser:
- http://192.168.1.67:8000/

## API Endpoints

- `GET /` - Main web interface
- `WebSocket /ws/video` - Video stream endpoint
- `GET /health` - Health check with client count
- `GET /api/stats` - Detailed streaming statistics

## Configuration

### Camera Settings

Edit `app/camera.py` to change resolution and framerate:

```python
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
```

### JPEG Quality

Adjust quality in `app/encoder.py`:

```python
DEFAULT_QUALITY = 95  # 0-100, higher is better
```

### Broadcast Timeout

Modify client timeout in `app/client_manager.py`:

```python
send_timeout = 0.02  # seconds
```

## Development

### Project Structure

```
app/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application
├── camera.py            # Camera capture module
├── encoder.py           # Frame encoder
├── client_manager.py    # WebSocket client manager
├── streaming.py         # Streaming service
├── static/              # Static assets
└── templates/           # HTML templates
    └── index.html
```

### Running Locally

```bash
# Inside the container
cd /app
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker

The application runs in a containerized environment with:
- NVIDIA L4T PyTorch base image
- CUDA/TensorRT support
- GStreamer with NVMM hardware acceleration

## Monitoring

Check logs:
```bash
ssh jetson@192.168.1.67
docker logs -f jetson-vision
```

View statistics:
```bash
curl http://192.168.1.67:8000/api/stats
```

## Troubleshooting

### Camera Not Detected

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

### Frame Freezing

- Check `/api/stats` for client count
- Verify network bandwidth
- Reduce JPEG quality if needed

### High Latency

- Ensure `appsink drop=true max-buffers=1` in pipeline
- Check client send timeout setting
- Monitor system resources with `tegrastats`

## License

MIT
