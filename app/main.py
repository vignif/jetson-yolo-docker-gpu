"""Main FastAPI application for Jetson camera streaming."""
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from streaming import StreamingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """WebSocket endpoint for video streaming.
    
    Clients connect here to receive real-time video frames.
    """
    await websocket.accept()
    streaming_service.client_manager.add_client(websocket)
    
    try:
        # Send latest frame immediately if available
        latest_frame = streaming_service.get_latest_frame()
        if latest_frame is not None:
            await websocket.send_bytes(latest_frame)
        
        # Keep connection alive
        # The streaming service broadcasts frames to all clients
        while True:
            await asyncio.sleep(1)
            # Could add ping/pong here if needed
    
    except WebSocketDisconnect:
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        streaming_service.client_manager.remove_client(websocket)


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
        "face_detection_backend": streaming_service.face_detector.get_backend(),
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


@app.post("/api/face-detection")
async def set_face_detection(request: FaceDetectionRequest):
    """Enable or disable face detection.
    
    Args:
        request: Face detection request with enabled flag
    """
    if request.enabled:
        streaming_service.face_detector.enable()
    else:
        streaming_service.face_detector.disable()
    
    return {
        "status": "ok",
        "enabled": streaming_service.face_detector.is_enabled()
    }


@app.get("/api/face-detection")
async def get_face_detection():
    """Get current face detection status."""
    return {
        "enabled": streaming_service.face_detector.is_enabled()
    }


# Import asyncio at the end to avoid circular imports
import asyncio
