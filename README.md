# Jetson Camera Streaming

Real-time camera streaming application for NVIDIA Jetson Nano with Raspberry Pi Camera Module v2.

## Features

- 🎥 **Hardware-accelerated video capture** via GStreamer with NVMM zero-copy
- 🧠 **Real-time object detection** with YOLOv5n (COCO 80 classes)
- 🎯 **Configurable detection** - select specific object classes to detect
- 🌐 **WebSocket-based streaming** for ultra-low latency (<100ms)
- 👥 **Multiple simultaneous viewers** support
- 🔧 **Live quality control** - adjust JPEG quality and detection confidence
- 📊 **GPU monitoring** with TensorRT and CUDA status
- 🌡️ **System telemetry** - temperature, CPU usage, power mode
- 🐳 **Fully containerized** with Docker for easy deployment
- ⚡ **GPU acceleration** with PyTorch CUDA support

## Architecture

The application is structured into modular components:

- **`camera.py`** - Camera capture with GStreamer pipeline management
- **`encoder.py`** - JPEG frame encoding with configurable quality
- **`client_manager.py`** - WebSocket client lifecycle and broadcasting
- **`streaming.py`** - Main streaming service orchestration
- **`main.py`** - FastAPI application and HTTP/WebSocket endpoints
- **`yolo_detector.py`** - YOLOv5n object detection with PyTorch (80 COCO classes)
- **`object_detector.py`** - Abstract detector interface
- **`system_monitor.py`** - System telemetry (temp, CPU, power)
- **`gpu_health_check.py`** - GPU validation and TensorRT checks
Prerequisites

- NVIDIA Jetson Nano with JetPack 4.x (L4T R32.x)
- Raspberry Pi Camera Module v2
- Docker and Docker Compose installed on Jetson

### Deploy to Jetson

```bash
# Deploy files from your computer to Jetson
./scripts/deploy.sh nvidia <JETSON-IP> jetson-webcam
# example ./scripts/deploy.sh nvidia 192.168.1.58 jetson-webcam

# SSH into Jetson and start the container
ssh nvidia@<JETSON-IP>
cd jetson-webcam
sudo docker-compose up -d --build
``` with live video stream
- `GET /gpu` - GPU status dashboard
- `WebSocket /ws/video` - Video stream endpoint
- `GET /health` - Health check with client count
- `GET /api/stats` - Detailed streaming statistics
- `GET /api/gpu-status` - GPU and TensorRT status
- `POST /api/run-gpu-tests` - Run GPU validation test
Open in your browser:
- **Main Stream**: http://<JETSON-IP>:8000/
- **GPU Status**: http://<JETSON-IP>:8000/gpu

Open in your browser:
- http://<JETSON-IP>:8000/

## API Endpoints

- `GET /` - Main web interface
- `WebSocket /ws/video` - Video stream endpoint
- `GET /health` - Health check with client count
- `GET /api/stats` - Detailed streaming statistics

## Object Detection

The application includes real-time object detection using YOLOv5n trained on the COCO dataset.

### Supported Classes (80 COCO objects)

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Indoor Objects**: chair, couch, bed, dining table, toilet, tv, laptop, keyboard, cell phone, book, clock

**Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, pizza

**Sports & Recreation**: frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

[See full list in app/yolo_detector.py]

### Configuration via Web UI

1. **Toggle Detection** - Enable/disable real-time detection
2. **Select Classes** - Choose which objects to detect (default: person only)
3. **Adjust Confidence** - Set detection threshold (default: 0.4)
4. **Quality Control** - Adjust JPEG quality for bandwidth optimization

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
├── __init__.py               # Package initialization
├── main.py                   # FastAPI application
├── camera.py                 # Camera capture module
├── encoder.py                # Frame encoder
├── client_manager.py         # WebSocket client manager
├── streaming.py              # Streaming service
├── yolo_detector.py          # YOLOv8 object detection
├── face_detector.py          # Face detection
├── face_detector_tensorrt.py # TensorRT-optimized face detection
├── object_detector.py        # Generic object detector interface
├── gpu_health_check.py       # GPU monitoring
├── system_monitor.py         # System resource monitoring
├── static/                   # Static assets
├── templates/                # HTML templates
│   ├── index.html           # Main streaming interface
│   └── gpu_status.html      # GPU dashboard
└── tests/                    # Unit and integration tests

scripts/
├── deploy.sh                 # Deploy to Jetson
├── rebuild.sh                # Rebuild container
├── run_tests.sh              # Run tests on device
├── check_gpu.py              # GPU diagnostics
├── test_imports.py           # Test TensorRT/CUDA imports
└── run_tests_local.py        # Local testing
```

### Running Locally

Inside the container:
```bash
cd /app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests

```bash
# On Jetson device
cd jetson-webcam
./scripts/run_tests.sh

# From development machine (runs tests remotely)
ssh nvidia@<JETSON-IP> "cd jetson-webcam && ./scripts/run_tests.sh"

# Or use docker-compose directly
sudo docker-compose exec jetson-vision python3 -m pytest tests/ -v
```

## Docker

The application runs in a containerized environment based on:
- **Base Image**: NVIDIA L4T PyTorch (r32.7.1-pth1.10-py3)
- **CUDA**: 10.2 with TensorRT 8.2.1.8
- **Hardware Acceleration**: GStreamer with NVMM zero-copy
- **Python**: 3.6 with FastAPI, OpenCV, and Ultralytics

### Building
### Check Container Logs
```bash
docker-compose logs -f jetson-vision
```

### View Statistics
```bash
curl http://<JETSON-IP>:8000/api/stats | python3 -m json.tool
```

### GPU Monitoring
```bash
# On Jetson
tegrastats

# Via API
# Check available cameras
ls -l /dev/video*
v4l2-ctl --list-devices

# Verify camera connection
dmesg | grep -i ov5647
```

### GPU Issues

```bash
# Run health check
docker-compose exec jetson-vision python3 /app/gpu_health_check.py

# Check CUDA
docker-compose exec jetson-vision nvcc --version

# View TensorRT version
docker-compose exec jetson-vision python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### Frame Freezing or High Latency

- Check `/api/stats` for client count and FPS
- Verify network bandwidth
- Reduce JPEG quality in encoder settings
- Monitor system resources: `tegrastats`
- Ensure camera framebuffer is not saturated

###Additional Documentation

- [Quick Reference](docs/QUICKSTART.md) - Common commands and URLs
- [YOLO Implementation](docs/YOLO_IMPLEMENTATION.md) - YOLOv8 integration details
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

See the [docs/](docs/) directory for more detailed documentation.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) before submitting a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built for NVIDIA Jetson Nano with JetPack 4.x
- Uses Raspberry Pi Camera Module v2 (IMX219)
- Powered by FastAPI, OpenCV, and Ultralytics YOLOv8nsure sufficient disk space (>10GB free)
- Check Docker version compatibility
- Verify JetPack version (4.x required)
- Review build logs for specific errors

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
