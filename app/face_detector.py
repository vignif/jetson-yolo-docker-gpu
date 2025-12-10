"""Face detection module using OpenCV Haar Cascades."""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """Handles face detection using Haar Cascade classifier."""
    
    def __init__(self):
        """Initialize face detector with Haar Cascade."""
        # Try multiple possible locations for Haar Cascade files
        possible_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
        ]
        
        # Try cv2.data if available
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                possible_paths.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            pass
        
        cascade_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break
        
        if cascade_path is None:
            raise RuntimeError(
                f"Could not find haarcascade_frontalface_default.xml in any of these locations: {possible_paths}"
            )
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")
        
        self.enabled = False
    
    def enable(self) -> None:
        """Enable face detection."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable face detection."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if face detection is enabled."""
        return self.enabled
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        if not self.enabled or frame is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes around detected faces.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            faces: List of face bounding boxes as (x, y, width, height) tuples
            
        Returns:
            Frame with bounding boxes drawn
        """
        if not faces:
            return frame
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            # Draw rectangle (green color, 2px thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optional: Add label
            cv2.putText(
                frame,
                'Face',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return frame
