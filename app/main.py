"""Main FastAPI application for Jetson camera streaming."""
import logging
import sys
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from streaming import StreamingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Run tests before starting application (only if RUN_TESTS env var is set)
if os.environ.get('RUN_TESTS', '').lower() == 'true':
    from run_tests import run_tests
    logger.info("Running test suite before application startup...")
    if not run_tests():
        logger.error("Tests failed! Application will not start.")
        sys.exit(1)
    logger.info("All tests passed! Starting application...")

# Initialize FastAPI app
app = FastAPI(
    title="Jetson Camera Stream",
    description="Real-time camera streaming with WebSocket support",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize streaming service (will be started on app startup)
streaming_service: StreamingService = StreamingService()


@app.on_event("startup")
async def on_startup():
    """Start streaming service when application starts."""
    logger.info("Application starting up...")
    await streaming_service.start()


@app.on_event("shutdown")
async def on_shutdown():
    """Stop streaming service when application shuts down."""
    logger.info("Application shutting down...")
    await streaming_service.stop()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    template_path = Path("templates/index.html")
    if not template_path.exists():
        return HTMLResponse(
            content="<h1>Template not found</h1>",
            status_code=500
        )
    
    with open(template_path) as f:
        return HTMLResponse(content=f.read())


@app.get("/gpu", response_class=HTMLResponse)
async def gpu_status_page():
    """Serve the GPU status dashboard page."""
    template_path = Path("templates/gpu_status.html")
    if not template_path.exists():
        return HTMLResponse(
            content="<h1>GPU Status template not found</h1>",
            status_code=500
        )
    
    with open(template_path) as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """WebSocket endpoint for video streaming.
    
    Clients connect here to receive real-time video frames.
    """
    client_id = id(websocket)
    logger.info("WebSocket connection attempt from client {}".format(client_id))
    
    await websocket.accept()
    logger.info("WebSocket accepted for client {}".format(client_id))
    streaming_service.client_manager.add_client(websocket)
    logger.info("Client {} added. Total: {}".format(client_id, streaming_service.get_client_count()))
    
    try:
        # Send latest frame immediately if available
        latest_frame = streaming_service.get_latest_frame()
        if latest_frame is not None:
            logger.info("Sending latest frame ({} bytes) to client {}".format(len(latest_frame), client_id))
            await websocket.send_bytes(latest_frame)
        else:
            logger.warning("No latest frame available for client {}".format(client_id))
        
        # Keep connection alive
        # The streaming service broadcasts frames to all clients
        while True:
            await asyncio.sleep(1)
            # Could add ping/pong here if needed
    
    except WebSocketDisconnect:
        logger.info("Client {} disconnected normally".format(client_id))
    except Exception as e:
        logger.error("WebSocket error for client {}: {}".format(client_id, e))
    finally:
        streaming_service.client_manager.remove_client(websocket)
        logger.info("Client {} removed. Remaining: {}".format(client_id, streaming_service.get_client_count()))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "clients": streaming_service.get_client_count(),
        "streaming": streaming_service.is_running
    }


@app.get("/api/stats")
async def stats():
    """Get streaming statistics."""
    memory = streaming_service.get_memory_usage()
    system_stats = streaming_service.system_monitor.get_all_stats()
    
    return {
        "connected_clients": streaming_service.get_client_count(),
        "is_streaming": streaming_service.is_running,
        "camera_resolution": f"{streaming_service.camera.width}x{streaming_service.camera.height}",
        "camera_fps": streaming_service.camera.fps,
        "actual_fps": streaming_service.get_fps(),
        "jpeg_quality": streaming_service.encoder.quality,
        "memory_mb": memory["used_mb"],
        "memory_percent": memory["percent"],
        "detection_enabled": streaming_service.is_detection_enabled(),
        "detection_backend": streaming_service.detector.get_backend_name(),
        "selected_classes_count": len(streaming_service.get_selected_classes()),
        "temperature": system_stats["temperature"],
        "fan_speed": system_stats["fan_speed"],
        "power_mode": system_stats["power_mode"],
        "cpu_usage": system_stats["cpu_usage"],
        "uptime": system_stats["uptime"]
    }


# Pydantic model for quality request
class QualityRequest(BaseModel):
    quality: int


# Pydantic model for face detection request
class FaceDetectionRequest(BaseModel):
    enabled: bool


# Pydantic model for object detection request
class ObjectDetectionRequest(BaseModel):
    enabled: bool


@app.post("/api/quality")
async def set_quality(request: QualityRequest):
    """Set JPEG encoding quality.
    
    Args:
        request: Quality request with value (0-100)
    """
    if not 0 <= request.quality <= 100:
        return {"error": "Quality must be between 0 and 100"}
    
    streaming_service.encoder.set_quality(request.quality)
    return {
        "status": "ok",
        "quality": streaming_service.encoder.quality
    }


@app.post("/api/detection")
async def set_detection(request: ObjectDetectionRequest):
    """Enable or disable detection.
    
    Args:
        request: Detection request with enabled flag
    """
    streaming_service.enable_detection(request.enabled)
    
    return {
        "status": "ok",
        "enabled": streaming_service.is_detection_enabled(),
        "backend": streaming_service.detector.get_backend_name()
    }


@app.get("/api/detection")
async def get_detection():
    """Get current detection status."""
    return {
        "enabled": streaming_service.is_detection_enabled(),
        "backend": streaming_service.detector.get_backend_name(),
        "selected_classes": streaming_service.get_selected_classes()
    }


@app.get("/api/detection/classes")
async def get_available_classes():
    """Get list of all available detection classes."""
    from yolo_detector import YOLOv5Detector
    return {
        "classes": YOLOv5Detector.get_available_classes()
    }


class ClassSelectionRequest(BaseModel):
    """Request model for selecting detection classes."""
    class_indices: list


@app.post("/api/detection/classes")
async def set_selected_classes(request: ClassSelectionRequest):
    """Set which object classes to detect.
    
    Args:
        request: List of class indices to detect
    """
    streaming_service.set_selected_classes(request.class_indices)
    
    return {
        "status": "ok",
        "selected_classes": streaming_service.get_selected_classes()
    }


class DetectionParamsRequest(BaseModel):
    """Request model for updating detection parameters."""
    conf_threshold: Optional[float] = None


@app.post("/api/detection/params")
async def set_detection_params(request: DetectionParamsRequest):
    """Update detection parameters.
    
    Args:
        request: Parameters to update (conf_threshold)
    """
    if request.conf_threshold is not None:
        streaming_service.detector.set_conf_threshold(request.conf_threshold)
    
    return {
        "status": "ok",
        "conf_threshold": streaming_service.detector.conf_threshold
    }


@app.get("/api/gpu-status")
async def gpu_status():
    """Get GPU availability and status."""
    import sys
    sys.path.insert(0, '/app')
    
    status = {
        "tensorrt_available": False,
        "pycuda_available": False,
        "cuda_device_available": False,
        "backend": streaming_service.face_detector.get_backend(),
        "gpu_accelerated": False
    }
    
    # Check TensorRT
    try:
        import tensorrt as trt
        status["tensorrt_available"] = True
        status["tensorrt_version"] = trt.__version__
    except:
        pass
    
    # Check PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        status["pycuda_available"] = True
        
        # Get device info
        device = cuda.Device(0)
        status["cuda_device_available"] = True
        status["gpu_name"] = device.name()
        status["compute_capability"] = f"{device.compute_capability()[0]}.{device.compute_capability()[1]}"
        status["total_memory_gb"] = round(device.total_memory() / (1024**3), 2)
    except:
        pass
    
    # Determine if GPU accelerated
    backend = status["backend"]
    status["gpu_accelerated"] = backend in ["TensorRT-GPU", "CUDA-DNN"]
    
    return status


@app.get("/api/run-gpu-tests")
async def run_gpu_tests():
    """Run GPU health checks and tests."""
    import subprocess
    import sys
    
    results = {
        "health_check": {"passed": False, "output": ""},
        "unit_tests": {"passed": False, "output": ""}
    }
    
    # Run health check
    try:
        result = subprocess.run(
            [sys.executable, "/app/gpu_health_check.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        results["health_check"]["output"] = result.stdout + result.stderr
        results["health_check"]["passed"] = result.returncode == 0
    except Exception as e:
        results["health_check"]["output"] = f"Error: {e}"
    
    # Run unit tests
    try:
        result = subprocess.run(
            [sys.executable, "/app/test_face_detection.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        results["unit_tests"]["output"] = result.stdout + result.stderr
        results["unit_tests"]["passed"] = result.returncode == 0
    except Exception as e:
        results["unit_tests"]["output"] = f"Error: {e}"
    
    return results


# Import asyncio at the end to avoid circular imports
import asyncio
