"""Camera capture module for Jetson with GStreamer pipeline."""
import cv2
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


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
        # Don't open camera in constructor - will open on first read
        self._initialized = False
    
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline for NVMM-accelerated capture."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1"
        )
    
    def _open(self) -> bool:
        """Open the video capture device.
        
        Returns:
            True if successfully opened, False otherwise
        """
        if self.cap is not None and self.cap.isOpened():
            return True
            
        try:
            logger.info("Opening camera with pipeline: {}".format(self.pipeline))
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self._initialized = True
                logger.info("Camera opened successfully")
                return True
            logger.error("Camera failed to open - isOpened() returned False")
            return False
        except Exception as e:
            logger.error("Exception opening camera: {}".format(e))
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
        """
        # Lazy initialization on first read
        if not self._initialized:
            if not self._open():
                return False, None
        
        if self.cap is None or not self.cap.isOpened():
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
