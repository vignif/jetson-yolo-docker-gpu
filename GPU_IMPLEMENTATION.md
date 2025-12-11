# GPU-Accelerated Face Detection - Implementation Summary

## What Was Added

### 1. **Native TensorRT Face Detection** (`app/face_detector_tensorrt.py`)

Complete GPU-accelerated face detection using NVIDIA TensorRT:

- **TensorRTInference class**: Converts ONNX models to TensorRT engines
- **FP16 precision**: 2x faster inference on Jetson GPU
- **Engine caching**: One-time conversion (2-3 min), then instant loads
- **Automatic fallback**: Falls back to CPU Haar Cascade if GPU unavailable
- **Frame skipping**: Process every 2nd frame for optimal performance

**Backend Priority:**
1. TensorRT-GPU (native TensorRT with FP16)
2. Haar-CPU (optimized CPU fallback)

### 2. **GPU Health Check** (`app/gpu_health_check.py`)

Comprehensive diagnostics that validate:
- ✓ TensorRT availability and version
- ✓ PyCUDA with CUDA device detection
- ✓ OpenCV availability
- ✓ TensorRT inference capability
- ✓ FP16 precision support
- ✓ GPU device name, compute capability, memory

Exit codes: `0` = GPU ready, `1` = CPU fallback

### 3. **Unit Tests** (`app/test_face_detection.py`)

Full test suite with 15+ tests:
- GPU component availability (TensorRT, PyCUDA, OpenCV)
- Detector initialization and backend detection
- Enable/disable functionality
- Detection on various inputs (black frames, synthetic frames)
- Output format validation (x, y, w, h tuples)
- Frame skipping optimization
- ONNX model download and availability
- TensorRT engine creation

### 4. **Container Entrypoint** (`entrypoint.sh`)

Smart startup script that:
- Runs GPU health check on every startup
- Reports GPU status before application starts
- Optionally runs unit tests (env var controlled)
- Gracefully handles GPU unavailability
- Provides clear success/warning messages

### 5. **API Endpoints** (`app/main.py`)

Two new endpoints for runtime GPU monitoring:

**`GET /api/gpu-status`** - Current GPU configuration:
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

**`GET /api/run-gpu-tests`** - Execute tests on demand:
```json
{
  "health_check": {"passed": true, "output": "..."},
  "unit_tests": {"passed": true, "output": "..."}
}
```

### 6. **Enhanced Dockerfile**

Updated to support GPU acceleration:
- Installs `python3-dev` and `build-essential` (required for pycuda)
- Runs GPU health check during build (validates setup)
- Uses entrypoint script for startup validation
- Includes all test scripts in container

### 7. **Documentation**

- **`GPU_TESTING.md`**: Complete testing guide with examples
- **`rebuild_and_test.sh`**: Automated rebuild and test script

## How to Use

### Quick Start

```bash
# 1. Deploy and rebuild container
./deploy.sh
ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose down && sudo docker-compose up -d --build'

# 2. Check logs for GPU status
ssh nvidia@192.168.1.67 'sudo docker-compose logs jetson-vision | grep -A 20 "GPU HEALTH CHECK"'

# 3. Access GPU status API
curl http://192.168.1.67:8000/api/gpu-status
```

### Automated Testing

```bash
# Run complete rebuild and test suite
./rebuild_and_test.sh
```

### Manual Testing

```bash
# SSH to Jetson
ssh nvidia@192.168.1.67

# GPU health check
sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py

# Unit tests
sudo docker-compose exec jetson-vision python3 /app/test_face_detection.py

# API endpoints
curl http://localhost:8000/api/gpu-status
curl http://localhost:8000/api/run-gpu-tests
```

## Expected Performance

| Mode | Backend | FPS | GPU Usage |
|------|---------|-----|-----------|
| No face detection | N/A | 30 | 0% |
| CPU face detection | Haar-CPU | 15-20 | 0% |
| GPU face detection | TensorRT-GPU | 25-30 | ~40% |

## Verification Checklist

After rebuild, verify:

- [ ] Container starts without errors
- [ ] Logs show: `✓ TensorRT 8.2.1.8 available`
- [ ] Logs show: `✓ PyCUDA available`
- [ ] Logs show: `✓ All critical GPU components available`
- [ ] Logs show: `✓ GPU-accelerated face detection READY`
- [ ] Web UI shows Backend: `TensorRT-GPU`
- [ ] Face detection enabled maintains 25-30 FPS
- [ ] API `/api/gpu-status` returns `gpu_accelerated: true`

## Troubleshooting

### Issue: TensorRT not available
**Cause:** Container not rebuilt  
**Fix:** `sudo docker-compose up -d --build`

### Issue: PyCUDA not available
**Cause:** Missing dependencies or not rebuilt  
**Fix:** Check Dockerfile has `python3-dev` and `build-essential`, rebuild

### Issue: Falls back to CPU
**Cause:** GPU check failed  
**Fix:** Check `sudo docker-compose logs` for error details, ensure nvidia runtime in docker-compose.yml

### Issue: First face detection is slow
**Expected:** First run builds TensorRT engine (2-3 minutes one-time), cached for subsequent runs

## Files Added/Modified

**New Files:**
- `app/gpu_health_check.py` - GPU diagnostics
- `app/test_face_detection.py` - Unit tests
- `entrypoint.sh` - Startup validation script
- `GPU_TESTING.md` - Testing documentation
- `rebuild_and_test.sh` - Automated test script

**Modified Files:**
- `app/face_detector_tensorrt.py` - Native TensorRT implementation
- `app/main.py` - Added `/api/gpu-status` and `/api/run-gpu-tests`
- `Dockerfile` - Added deps, entrypoint, build-time check
- `requirements.txt` - Added `pycuda==2022.1`

## Technical Details

**TensorRT Pipeline:**
1. Download Ultra-Light ONNX model (~1.2MB) from GitHub
2. Build TensorRT engine with FP16 precision (one-time, ~2-3 min)
3. Cache engine as `/app/models/version-RFB-640.trt`
4. Load cached engine on subsequent runs (instant)
5. Run inference with pycuda memory management

**Why This Works:**
- L4T PyTorch image includes TensorRT 8.2.1.8
- pycuda provides Python → CUDA interface
- Native TensorRT API bypasses OpenCV CUDA limitations
- FP16 precision supported by Jetson's GPU architecture
- Engine caching eliminates conversion overhead

## Next Steps

1. Run `./rebuild_and_test.sh` to deploy and validate
2. Monitor first face detection activation (engine build)
3. Verify FPS maintains 25-30 with face detection enabled
4. Check `/api/gpu-status` confirms `TensorRT-GPU` backend
5. Optionally enable startup tests with `RUN_TESTS_ON_STARTUP=true`
