"""Video frame encoder with JPEG compression."""
import cv2
from typing import Optional, Tuple
import numpy as np


class FrameEncoder:
    """Handles frame encoding to JPEG format."""
    
    DEFAULT_QUALITY = 95
    
    def __init__(self, quality: int = DEFAULT_QUALITY):
        """Initialize frame encoder.
        
        Args:
            quality: JPEG quality (0-100, higher is better)
        """
        self.quality = max(0, min(100, quality))
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
    
    def encode(self, frame: np.ndarray) -> Tuple[bool, Optional[bytes]]:
        """Encode a frame to JPEG format.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (success: bool, jpeg_bytes: Optional[bytes])
        """
        if frame is None or frame.size == 0:
            return False, None
        
        success, buffer = cv2.imencode('.jpg', frame, self.encode_params)
        if not success:
            return False, None
        
        return True, buffer.tobytes()
    
    def set_quality(self, quality: int) -> None:
        """Update JPEG quality setting.
        
        Args:
            quality: New JPEG quality (0-100)
        """
        self.quality = max(0, min(100, quality))
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
