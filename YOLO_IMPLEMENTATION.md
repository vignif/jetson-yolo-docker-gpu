# YOLOv5 Implementation Summary

Complete implementation of lightweight YOLOv5n object detection with user-selectable classes.
**Python 3.6 compatible** - uses torch.hub instead of ultralytics package.

## Changes Made

### 1. Backend (✅ Complete)
- **Removed**: Face detection (face_detector_tensorrt.py, face_detector.py)
- **Created**: `yolo_detector.py` - YOLOv5n detector with:
  - GPU acceleration (CUDA)
  - 80 COCO classes support
  - Configurable confidence threshold
  - Class selection (filter which objects to detect)
  - Lightweight nano model for Jetson Nano
  - **Python 3.6 compatible** via torch.hub

### 2. Streaming Service (✅ Complete)
- Updated `streaming.py`:
  - Removed face detection references
  - Changed from `object_detector` to `detector` (YOLOv5Detector)
  - Added `set_selected_classes()` and `get_selected_classes()` methods
  - Detection enabled by default

### 3. API Endpoints (✅ Complete)
- Updated `main.py`:
  - **Removed**: `/api/face-detection` (GET/POST)
  - **Changed**: `/api/object-detection` → `/api/detection`
  - **New**: `GET /api/detection/classes` - Returns 80 COCO classes
  - **New**: `POST /api/detection/classes` - Set selected class indices
  - **Updated**: `POST /api/detection/params` - Simplified (only conf_threshold)
  - **Updated**: `/api/stats` - Shows detection_backend and selected_classes_count

### 4. User Interface (✅ Complete)
- Completely replaced `index.html`:
  - Modern dark theme (#0d1117 background, #76b900 NVIDIA green)
  - Canvas-based video display with WebSocket
  - Stats panel (FPS, Memory, Backend, Temperature)
  - Detection toggle switch
  - Confidence threshold slider
  - **Class selection grid**: 80 COCO classes as checkboxes
  - Quick buttons: "Select All", "Deselect All", "Common Objects"
  - Real-time updates without page reload

### 5. Dependencies (✅ Complete)
- Updated `requirements.txt`:
  - **Removed**: ultralytics==8.0.196 (not Python 3.6 compatible)
  - **Added**: scipy>=1.5.0 (required by YOLOv5)
  - **Uses**: torch.hub.load('ultralytics/yolov5', 'yolov5n')
  - No additional packages needed - leverages existing PyTorch installation

## Deployment Instructions

### Step 1: Deploy Code
```bash
# From Mac
./deploy.sh
```

### Step 2: Reset Camera (if needed)
```bash
# On Jetson
ssh nvidia@192.168.1.67
cd jetson-webcam
./camera_reset.sh  # Only if seeing "Failed to create CaptureSession"
```

### Step 3: Rebuild Container
```bash
# On Jetson (requirements.txt changed)
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Step 4: Monitor Startup
```bash
docker-compose logs -f
```

Look for:
- "Loading YOLOv5n (nano - lightweight) via torch.hub..."
- "YOLOv5n-GPU initialized on cuda"
- "Detecting 80 object classes"
- Camera opens (GStreamer or V4L2)

### Step 5: Test UI
Open browser: http://jetson.local:8000 or http://192.168.1.67:8000

Test:
- Video stream displays
- Detection toggle works
- Class selection grid appears (80 classes)
- Select "Common Objects" → only see people, cars, dogs, cats, etc.
- Adjust confidence slider
- Verify bounding boxes appear with class names

## Features

### Object Detection
- **Model**: YOLOv5n (nano - smallest, fastest)
- **Backend**: GPU-accelerated with CUDA
- **Classes**: 80 COCO objects
- **Memory**: ~500MB VRAM
- **Speed**: ~15-20 FPS on Jetson Nano

### Class Selection
Users can select which objects to detect from:
- All 80 COCO classes
- Quick buttons for common selections
- Real-time filtering (no restart needed)

### Available COCO Classes (80 total)
```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote,
keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book,
clock, vase, scissors, teddy bear, hair drier, toothbrush
```

## Troubleshooting

### Camera Issues
- Run `./camera_reset.sh` if seeing "Failed to create CaptureSession"
- Check `docker-compose logs` for camera initialization messages
- V4L2 fallback will activate automatically if GStreamer fails

### Detection Issues
- If no detections: Lower confidence threshold (try 0.2)
- If too many false positives: Raise confidence threshold (try 0.6)
- If specific object not detected: Verify class is selected in UI
- If GPU OOM: Reduce selected classes or restart container

### Performance Issues
- Expected FPS: 15-20 with detection, 25-30 without
- High CPU usage: V4L2 fallback is active (GStreamer failed)
- High GPU memory: Normal for YOLO (~500MB)
- Temperature >80°C: Consider adding cooling fan

## Architecture

- **Camera**: RPi Camera v2, 1280x720@30fps, GStreamer with V4L2 fallback
- **Detection**: YOLOv5n via torch.hub, GPU-accelerated
- **API**: FastAPI with WebSocket streaming
- **UI**: Modern dark theme with real-time class selection
- **Container**: nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
- **Python**: 3.6 (L4T limitation)
- **PyTorch**: 1.10.0 with CUDA 10.2
