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
        self.use_gstreamer = True  # Try GStreamer first
    
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
        
        # Try GStreamer nvarguscamerasrc first
        if self.use_gstreamer:
            try:
                logger.info("Attempting GStreamer pipeline: {}".format(self.pipeline))
                self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
                
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    logger.info("GStreamer camera opened, testing frame read...")
                    import time
                    time.sleep(1)  # Give camera time to initialize
                    
                    test_success, test_frame = self.cap.read()
                    if test_success and test_frame is not None:
                        self._initialized = True
                        logger.info("✓ Camera opened successfully with GStreamer")
                        return True
                    else:
                        logger.warning("GStreamer opened but cannot read frames, releasing...")
                        self.cap.release()
                        self.cap = None
                        self.use_gstreamer = False
            except Exception as e:
                logger.error("GStreamer exception: {}".format(e))
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.use_gstreamer = False
        
        # Fallback to direct V4L2 access
        logger.info("Falling back to V4L2 direct access...")
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Test read
                test_success, test_frame = self.cap.read()
                if test_success and test_frame is not None:
                    self._initialized = True
                    logger.info("✓ Camera opened successfully with V4L2")
                    logger.info("  Resolution: {}x{}, FPS: {}".format(
                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.cap.get(cv2.CAP_PROP_FPS))
                    ))
                    return True
                else:
                    logger.error("V4L2 opened but cannot read frames")
                    self.cap.release()
                    self.cap = None
            else:
                logger.error("Failed to open V4L2 camera")
        except Exception as e:
            logger.error("V4L2 exception: {}".format(e))
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
        """
        # Lazy initialization on first read
        if not self._initialized:
            logger.info("Camera not initialized, attempting to open...")
            if not self._open():
                logger.error("Failed to open camera on first read")
                return False, None
        
        if self.cap is None:
            logger.error("Camera capture object is None")
            return False, None
            
        if not self.cap.isOpened():
            logger.error("Camera is not opened")
            return False, None
            
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame from camera")
        return success, frame
    
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
