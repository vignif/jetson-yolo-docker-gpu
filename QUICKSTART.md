# Quick Reference - GPU Face Detection

## URLs

- **Live Stream**: http://192.168.1.67:8000/
- **GPU Status Dashboard**: http://192.168.1.67:8000/gpu
- **API - GPU Status**: http://192.168.1.67:8000/api/gpu-status
- **API - Run Tests**: http://192.168.1.67:8000/api/run-gpu-tests
- **API - System Stats**: http://192.168.1.67:8000/api/stats

## Rebuild Container (Required First Time)

```bash
ssh nvidia@192.168.1.67
cd jetson-webcam
sudo docker-compose down
sudo docker-compose build --no-cache
sudo docker-compose up -d
```

**Important**: Use `--no-cache` to ensure clean build with system pycuda

**Wait 5-10 minutes** for build to complete

## Check GPU Status

### Via Browser
Open: http://192.168.1.67:8000/gpu

### Via Command Line
```bash
# Health check
ssh nvidia@192.168.1.67 'sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py'

# Unit tests  
ssh nvidia@192.168.1.67 'sudo docker-compose exec jetson-vision python3 /app/test_face_detection.py'

# API
curl http://192.168.1.67:8000/api/gpu-status | python3 -m json.tool
```

## Expected Output (Success)

```
✓ TensorRT 8.2.1.8 available
✓ PyCUDA available
✓ GPU Device: NVIDIA Tegra X1
✓ Compute Capability: (5, 3)
✓ Total Memory: 3.87 GB
✓ All critical GPU components available
✓ GPU-accelerated face detection READY
```

**Backend**: `TensorRT-GPU`  
**Performance**: 25-30 FPS with face detection enabled

## First Face Detection Activation

**Important**: First time you enable face detection:
1. Downloads ONNX model (~1.2MB) - takes 10 seconds
2. Builds TensorRT engine - **takes 2-3 minutes**
3. Caches engine - subsequent runs are instant

Monitor progress:
```bash
ssh nvidia@192.168.1.67 'sudo docker-compose logs -f jetson-vision'
```

Look for:
```
Downloading optimized face detection model...
✓ Model downloaded
Building TensorRT engine from /app/models/version-RFB-640.onnx...
✓ Using FP16 precision for faster inference
✓ TensorRT engine saved to /app/models/version-RFB-640.trt
✓ Native TensorRT inference ready (GPU accelerated)
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
ssh nvidia@192.168.1.67 'sudo docker-compose logs jetson-vision'

# Rebuild from scratch
ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose down -v && sudo docker-compose up -d --build'
```

### GPU Not Detected
```bash
# Verify nvidia runtime
ssh nvidia@192.168.1.67 'grep -A 5 "runtime: nvidia" jetson-webcam/docker-compose.yml'

# Check GPU on host
ssh nvidia@192.168.1.67 'nvidia-smi' # or 'tegrastats'
```

### Falls Back to CPU
Check entrypoint logs at container startup:
```bash
ssh nvidia@192.168.1.67 'sudo docker-compose logs jetson-vision | head -50'
```

### PyCUDA Not Found
Rebuild required:
```bash
ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose up -d --build'
```

## Files Overview

| File | Purpose |
|------|---------|
| `app/face_detector_tensorrt.py` | TensorRT face detection |
| `app/gpu_health_check.py` | GPU diagnostics |
| `app/test_face_detection.py` | Unit tests |
| `entrypoint.sh` | Startup validation |
| `app/templates/gpu_status.html` | GPU dashboard UI |
| `GPU_TESTING.md` | Full testing guide |
| `GPU_IMPLEMENTATION.md` | Technical details |

## Test Commands

```bash
# Automated rebuild and test
./rebuild_and_test.sh

# Manual tests
ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py'
ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose exec jetson-vision python3 /app/test_face_detection.py'

# Web-based tests
# Open http://192.168.1.67:8000/gpu and click "Run All Tests"
```

## Performance Expectations

| Scenario | Backend | FPS | Notes |
|----------|---------|-----|-------|
| No detection | - | 30 | Baseline |
| GPU detection | TensorRT-GPU | 25-30 | Optimal |
| CPU detection | Haar-CPU | 15-20 | Fallback |

## Next Steps

1. ✅ Deploy: `./deploy.sh`
2. ✅ Rebuild: `ssh nvidia@192.168.1.67 'cd jetson-webcam && sudo docker-compose up -d --build'`
3. ✅ Wait for build (5-10 min)
4. ✅ Check GPU dashboard: http://192.168.1.67:8000/gpu
5. ✅ Enable face detection in main UI
6. ✅ Wait for TensorRT engine build (2-3 min, first time only)
7. ✅ Verify 25-30 FPS maintained

## Support

For issues:
1. Check GPU dashboard: http://192.168.1.67:8000/gpu
2. Review logs: `sudo docker-compose logs jetson-vision`
3. Run health check: `sudo docker-compose exec jetson-vision python3 /app/gpu_health_check.py`
4. Consult `GPU_TESTING.md` for detailed troubleshooting
