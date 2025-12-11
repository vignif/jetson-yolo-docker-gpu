"""Optimized face detection for Jetson (CPU-based with future GPU support)."""
import cv2
import numpy as np
from typing import List, Tuple


class FaceDetectorTensorRT:
    """Optimized face detection for Jetson Nano.
    
    Note: TensorRT requires pycuda which has compatibility issues with L4T images.
    Using optimized Haar Cascade for reliable CPU-based detection.
    Future: Can be upgraded to use TensorRT C++ API or DeepStream SDK.
    """
    
    def __init__(self):
        """Initialize face detector."""
        self.enabled = False
        self.backend = "Haar-CPU-Optimized"
        self.last_faces = []
        self.frame_count = 0
        self.frame_skip = 3  # Process every 3rd frame for performance
        
        print("Initializing optimized face detection...")
        self._init_haar_detector()
        print(f"✓ Face detector ready with {self.backend} backend")
    
    def _init_haar_detector(self) -> None:
        """Initialize optimized Haar Cascade detector."""
        cascade_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            try:
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    print(f"✓ Loaded Haar Cascade from {path}")
                    self.detection_method = "haar"
                    self.backend = "Haar-CPU-Optimized"
                    return
            except Exception as e:
                continue
        
        raise RuntimeError("Failed to load Haar Cascade classifier from any path")
    
    def enable(self) -> None:
        """Enable face detection."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable face detection."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if face detection is enabled."""
        return self.enabled
    
    def get_backend(self) -> str:
        """Get current backend."""
        return self.backend
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        if not self.enabled or frame is None:
            return []
        
        self.frame_count += 1
        
        # Frame skipping for performance
        if self.frame_count % self.frame_skip != 0:
            return self.last_faces
        
        try:
            faces = self._detect_haar(frame)
            self.last_faces = faces
            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return self.last_faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade (optimized for Jetson).
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of (x, y, w, h) tuples
        """
        h, w = frame.shape[:2]
        
        if h <= 0 or w <= 0:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize: resize for detection (faster processing)
        detection_scale = 0.5
        small_h = int(h * detection_scale)
        small_w = int(w * detection_scale)
        small_gray = cv2.resize(gray, (small_w, small_h))
        
        # Detect faces with optimized parameters for Jetson
        faces = self.face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.15,  # Slightly aggressive for speed
            minNeighbors=4,     # Reduce false positives
            minSize=(40, 40),   # Minimum face size in scaled image
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Scale back to original coordinates
        scaled_faces = []
        for (x, y, w, h) in faces:
            x = int(x / detection_scale)
            y = int(y / detection_scale)
            w = int(w / detection_scale)
            h = int(h / detection_scale)
            
            # Clamp to frame bounds
            x = max(0, min(x, frame.shape[1]))
            y = max(0, min(y, frame.shape[0]))
            w = max(0, min(w, frame.shape[1] - x))
            h = max(0, min(h, frame.shape[0] - y))
            
            if w > 0 and h > 0:
                scaled_faces.append((x, y, w, h))
        
        return scaled_faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw rectangles around detected faces.
        
        Args:
            frame: Input BGR image
            faces: List of (x, y, w, h) tuples for detected faces
            
        Returns:
            Frame with drawn rectangles
        """
        if faces is None or len(faces) == 0:
            return frame
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            # Draw rectangle with green color and 2px thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optional: Add label
            label = "Face"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_y = max(y - 10, label_size[1])
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
