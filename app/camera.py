"""Camera capture module for Jetson with GStreamer pipeline."""
import cv2
from typing import Optional, Tuple
import numpy as np


class CameraCapture:
    """Handles video capture from Raspberry Pi Camera Module via GStreamer."""
    
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_FPS = 30
    
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, fps: int = DEFAULT_FPS):
        """Initialize camera capture with specified resolution and framerate.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = self._build_pipeline()
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline for NVMM-accelerated capture."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1"
        )
    
    def _open(self) -> None:
        """Open the video capture device."""
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera capture")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def is_opened(self) -> bool:
        """Check if camera is opened and ready."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
