from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import cv2

app = FastAPI(title="Jetson Camera Stream")

app.mount("/static", StaticFiles(directory="static"), name="static")

current_model = "none"

class CameraCapture:
    def __init__(self):
        # Match the working pipeline caps: 1280x720 @30fps, NVMM buffers
        self.pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

@app.get("/")
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    cam = CameraCapture()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                await asyncio.sleep(0.05)
                continue
            # Preserve native resolution; do not resize
            # Higher JPEG quality to avoid artifacts
            ok_enc, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok_enc:
                await asyncio.sleep(0)
                continue
            # Send as binary to reduce overhead
            await websocket.send_bytes(buf.tobytes())
            # Yield to event loop without pacing to 30fps artificially
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        cam.release()

@app.get("/health")
async def health():
    return {"status": "ok"}
