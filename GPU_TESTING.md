# GPU Testing and Validation

This document describes the GPU acceleration testing and validation implemented in the Jetson webcam streaming application.

## Overview

The application uses NVIDIA TensorRT for GPU-accelerated face detection on the Jetson Nano. Multiple layers of testing ensure GPU functionality works correctly in the Docker container.

## Testing Components

### 1. GPU Health Check (`app/gpu_health_check.py`)

Comprehensive diagnostic script that validates all GPU components:

- ✓ TensorRT availability and version
- ✓ PyCUDA availability
- ✓ CUDA device detection and info
- ✓ OpenCV availability
- ✓ TensorRT inference capability test
- ✓ FP16 precision support check

**Run manually:**
```bash
# Inside container
python3 /app/gpu_health_check.py

# From host
sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py
```

**Exit codes:**
- `0` = All critical components available (GPU ready)
- `1` = Missing components (CPU fallback will be used)

### 2. Unit Tests (`app/test_face_detection.py`)

Automated unit tests for face detection functionality:

**Test Classes:**

- `TestGPUAvailability` - Tests TensorRT, PyCUDA, OpenCV imports
- `TestFaceDetectorTensorRT` - Tests detector initialization, enable/disable, detection on various inputs
- `TestTensorRTInference` - Tests ONNX model download and TensorRT engine creation

**Test Coverage:**
- ✓ Component imports and availability
- ✓ Detector initialization
- ✓ GPU backend detection
- ✓ Enable/disable functionality
- ✓ Detection on black frames
- ✓ Detection on synthetic frames
- ✓ Output format validation
- ✓ Frame skipping optimization
- ✓ ONNX model availability
- ✓ TensorRT engine creation

**Run manually:**
```bash
# Inside container
python3 /app/test_face_detection.py

# From host
sudo docker-compose exec jetson-vision python3 /app/test_face_detection.py
```

### 3. Container Entrypoint (`entrypoint.sh`)

Validates GPU availability on every container startup:

- Runs GPU health check automatically
- Reports GPU status before starting application
- Optional unit test execution (controlled by env vars)
- Graceful fallback to CPU if GPU unavailable

**Environment Variables:**
- `RUN_TESTS_ON_STARTUP=true` - Run unit tests on startup (default: false)
- `FAIL_ON_TEST_ERROR=true` - Exit if tests fail (default: false)

### 4. API Endpoints

Runtime GPU status and testing via HTTP API:

#### `GET /api/gpu-status`

Returns current GPU availability and configuration:

```json
{
  "tensorrt_available": true,
  "tensorrt_version": "8.2.1.8",
  "pycuda_available": true,
  "cuda_device_available": true,
  "gpu_name": "NVIDIA Tegra X1",
  "compute_capability": "5.3",
  "total_memory_gb": 3.87,
  "backend": "TensorRT-GPU",
  "gpu_accelerated": true
}
```

#### `GET /api/run-gpu-tests`

Runs health check and unit tests on demand:

```json
{
  "health_check": {
    "passed": true,
    "output": "..."
  },
  "unit_tests": {
    "passed": true,
    "output": "..."
  }
}
```

## Usage Examples

### Check GPU Status from Browser

1. Navigate to `http://192.168.1.67:8000/api/gpu-status`
2. Verify `gpu_accelerated: true` and `backend: "TensorRT-GPU"`

### Run Tests via API

```bash
curl http://192.168.1.67:8000/api/run-gpu-tests
```

### Run Tests During Container Build

Tests run automatically during `docker-compose build`:

```bash
sudo docker-compose build
```

### Run Tests on Container Startup

```bash
# Enable test execution on startup
docker-compose up -e RUN_TESTS_ON_STARTUP=true

# Fail container start if tests fail
docker-compose up -e RUN_TESTS_ON_STARTUP=true -e FAIL_ON_TEST_ERROR=true
```

### Manual Test Execution

```bash
# SSH to Jetson
ssh nvidia@192.168.1.67

# Run health check
sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py

# Run unit tests
sudo docker-compose exec jetson-vision python3 /app/test_face_detection.py

# Check GPU status via API
curl http://localhost:8000/api/gpu-status
```

## Expected Results

### With GPU Acceleration (TensorRT)

**Health Check Output:**
```
✓ TensorRT 8.2.1.8 available
✓ PyCUDA available
✓ GPU Device: NVIDIA Tegra X1
✓ Compute Capability: (5, 3)
✓ Total Memory: 3.87 GB
✓ TensorRT runtime functional
✓ FP16 precision supported
✓ All critical GPU components available
✓ GPU-accelerated face detection READY
```

**Backend:** `TensorRT-GPU`  
**FPS with Face Detection:** 25-30 FPS

### Without GPU (CPU Fallback)

**Health Check Output:**
```
✗ PyCUDA not available
⚠ Some GPU components missing
⚠ Will fall back to CPU-based detection
```

**Backend:** `Haar-CPU` or `CPU`  
**FPS with Face Detection:** 15-20 FPS

## Troubleshooting

### TensorRT Import Fails

**Issue:** `import tensorrt` fails  
**Solution:** TensorRT is included in L4T PyTorch base image, should work by default

### PyCUDA Import Fails

**Issue:** `import pycuda` fails  
**Solution:** Rebuild container to install pycuda:
```bash
cd jetson-webcam
sudo docker-compose down
sudo docker-compose up -d --build
```

**Requirements:** 
- `python3-dev` (installed in Dockerfile)
- `build-essential` (installed in Dockerfile)
- `pycuda==2022.1` (in requirements.txt)

### CUDA Device Not Found

**Issue:** `getCudaEnabledDeviceCount() = 0`  
**Solution:** Ensure docker-compose.yml uses `nvidia` runtime and `/dev/video0` is mapped

### Tests Fail on Non-GPU Host

**Expected:** Building on x86 machine without GPU will show:
```
GPU check failed (expected if building on non-GPU host)
```

This is normal - tests will pass when container runs on Jetson with GPU.

## CI/CD Integration

For automated testing in deployment pipelines:

```bash
#!/bin/bash
# deployment_test.sh

# Build and start container
sudo docker-compose up -d --build

# Wait for startup
sleep 10

# Run health check
sudo docker-compose exec -T jetson-vision python3 /app/gpu_health_check.py
if [ $? -ne 0 ]; then
    echo "ERROR: GPU health check failed"
    exit 1
fi

# Run unit tests
sudo docker-compose exec -T jetson-vision python3 /app/test_face_detection.py
if [ $? -ne 0 ]; then
    echo "ERROR: Unit tests failed"
    exit 1
fi

# Check API endpoint
STATUS=$(curl -s http://localhost:8000/api/gpu-status | jq -r '.gpu_accelerated')
if [ "$STATUS" != "true" ]; then
    echo "ERROR: GPU acceleration not active"
    exit 1
fi

echo "✓ All tests passed - GPU acceleration confirmed"
```

## Performance Metrics

| Configuration | Backend | Face Detection FPS | Notes |
|--------------|---------|-------------------|-------|
| TensorRT GPU | TensorRT-GPU | 25-30 FPS | Optimal performance |
| CUDA DNN | CUDA-DNN | 20-25 FPS | Fallback if TensorRT fails |
| CPU Haar | Haar-CPU | 15-20 FPS | CPU-only fallback |
| No Detection | N/A | 30 FPS | Baseline without face detection |

## Architecture

```
Container Startup
    ↓
entrypoint.sh
    ↓
gpu_health_check.py → Validates GPU components
    ↓
[Optional] test_face_detection.py → Runs unit tests
    ↓
Application Start (uvicorn)
    ↓
face_detector_tensorrt.py → Initializes with GPU or CPU fallback
    ↓
Runtime: /api/gpu-status, /api/run-gpu-tests available
```

## Files

- `app/gpu_health_check.py` - GPU diagnostics script
- `app/test_face_detection.py` - Unit tests
- `app/face_detector_tensorrt.py` - TensorRT face detector implementation
- `entrypoint.sh` - Container startup with validation
- `Dockerfile` - Includes GPU check during build
- `app/main.py` - API endpoints for GPU status and testing
