from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Set, Optional
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

# Global capture loop and client management
clients: Set[WebSocket] = set()
latest_jpeg: Optional[bytes] = None
capture_task: Optional[asyncio.Task] = None

async def capture_loop():
    global latest_jpeg
    cam = CameraCapture()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue
            ok_enc, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok_enc:
                await asyncio.sleep(0)
                continue
            latest_jpeg = buf.tobytes()
            # Broadcast to connected clients; drop slow ones
            if clients:
                send_tasks = []
                for ws in list(clients):
                    try:
                        # Skip if connection is closed
                        if ws.application_state.name != 'CONNECTED':
                            clients.discard(ws)
                            continue
                        # Send with small timeout to avoid blocking capture
                        send_tasks.append(asyncio.wait_for(ws.send_bytes(latest_jpeg), timeout=0.02))
                    except Exception:
                        clients.discard(ws)
                if send_tasks:
                    # Gather with exceptions; do not raise
                    try:
                        await asyncio.gather(*send_tasks, return_exceptions=True)
                    except Exception:
                        pass
            await asyncio.sleep(0)
    finally:
        cam.release()

@app.on_event("startup")
async def on_startup():
    global capture_task
    if capture_task is None:
        capture_task = asyncio.create_task(capture_loop())

@app.get("/")
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        # On connect, send the latest frame immediately if available
        if latest_jpeg is not None:
            await websocket.send_bytes(latest_jpeg)
        # Keep the connection open; frames are pushed by capture_loop
        while True:
            # Ping/pong handling could be added; for now just sleep
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            clients.remove(websocket)
        except KeyError:
            pass

@app.get("/health")
async def health():
    return {"status": "ok"}
