from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import cv2
import base64

app = FastAPI(title="Jetson Camera Stream")

app.mount("/static", StaticFiles(directory="static"), name="static")

current_model = "none"

class CameraCapture:
    def __init__(self):
        self.pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
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
            frame = cv2.resize(frame, (640, 480))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode('utf-8')
            await websocket.send_json({"frame": b64})
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        pass
    finally:
        cam.release()

@app.get("/health")
async def health():
    return {"status": "ok"}
