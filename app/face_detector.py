"""Face detection module using OpenCV with GPU acceleration."""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """Handles face detection using DNN with CUDA acceleration when available."""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize face detector with GPU acceleration if available.
        
        Args:
            use_gpu: Whether to attempt using GPU acceleration
        """
        self.enabled = False
        self.use_gpu = use_gpu
        self.backend = "CPU"
        self.detection_method = "haar"
        
        # Performance optimization settings
        self.frame_skip = 3  # Detect faces every N frames
        self.frame_count = 0
        self.last_faces = []  # Cache last detection results
        self.detection_scale = 0.5  # Scale factor for detection (0.5 = half size)
        
        # Try GPU-accelerated DNN first
        if use_gpu:
            try:
                dnn_initialized = self._init_dnn_detector()
                if dnn_initialized:
                    return
            except Exception as e:
                print(f"Failed to initialize GPU face detector: {e}")
        
        # Fallback to Haar Cascade (CPU)
        print("Using Haar Cascade face detection (CPU)")
        self._init_haar_detector()
    
    def _init_dnn_detector(self) -> bool:
        """Initialize DNN-based face detection with GPU support.
        
        Returns:
            True if successful, False otherwise
        """
        model_path = self._get_dnn_model()
        config_path = self._get_dnn_config()
        
        if not model_path or not config_path:
            return False
        
        print(f"Loading DNN model from {model_path}")
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        
        # Try to set CUDA backend
        try:
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount() if cuda_available else 0}")
        except Exception as e:
            print(f"CUDA check failed: {e}")
            cuda_available = False
        
        if cuda_available:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.backend = "CUDA"
                print("✓ Face detection using CUDA backend")
            except Exception as e:
                print(f"Failed to set CUDA backend: {e}")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "DNN-CPU"
                print("✓ Face detection using DNN CPU backend (optimized)")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "DNN-CPU"
            print("✓ Face detection using DNN CPU backend (optimized)")
        
        self.detection_method = "dnn"
        self.confidence_threshold = 0.5
        # Use smaller detection size for DNN to improve speed
        self.dnn_input_size = 160  # Reduced from 300 for faster inference
        return True
    
    def _init_haar_detector(self) -> None:
        """Initialize Haar Cascade face detection (CPU fallback)."""
        cascade_path = self._find_haar_cascade()
        if cascade_path is None:
            raise RuntimeError("Could not find Haar Cascade file")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path}")
        
        self.detection_method = "haar"
        self.backend = "CPU"
    
    def _get_dnn_model(self) -> Optional[str]:
        """Get or download DNN face detection model."""
        model_file = "/app/models/res10_300x300_ssd_iter_140000.caffemodel"
        
        if os.path.exists(model_file):
            return model_file
        
        os.makedirs("/app/models", exist_ok=True)
        
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            print(f"Downloading face detection model (~10MB)...")
            urllib.request.urlretrieve(url, model_file)
            print(f"✓ Model downloaded to {model_file}")
            return model_file
        except Exception as e:
            print(f"Failed to download model: {e}")
            return None
    
    def _get_dnn_config(self) -> Optional[str]:
        """Get or download DNN model config."""
        config_file = "/app/models/deploy.prototxt"
        
        if os.path.exists(config_file):
            return config_file
        
        os.makedirs("/app/models", exist_ok=True)
        
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            print(f"Downloading model config...")
            urllib.request.urlretrieve(url, config_file)
            print(f"✓ Config downloaded to {config_file}")
            return config_file
        except Exception as e:
            print(f"Failed to download config: {e}")
            return None
    
    def _find_haar_cascade(self) -> Optional[str]:
        """Find Haar Cascade XML file."""
        possible_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
        ]
        
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                possible_paths.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
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
        """Get current backend being used."""
        return self.backend
    
    def set_performance_mode(self, mode: str) -> None:
        """Set performance optimization mode.
        
        Args:
            mode: 'fast' (skip more frames, lower res) or 'accurate' (skip fewer frames, higher res)
        """
        if mode == "fast":
            self.frame_skip = 5
            self.detection_scale = 0.4
            self.dnn_input_size = 128
            print("Face detection: FAST mode (5 frame skip, 0.4x scale)")
        elif mode == "accurate":
            self.frame_skip = 2
            self.detection_scale = 0.75
            self.dnn_input_size = 300
            print("Face detection: ACCURATE mode (2 frame skip, 0.75x scale)")
        else:  # balanced
            self.frame_skip = 3
            self.detection_scale = 0.5
            self.dnn_input_size = 160
            print("Face detection: BALANCED mode (3 frame skip, 0.5x scale)")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame with frame skipping optimization.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        if not self.enabled or frame is None:
            return []
        
        self.frame_count += 1
        
        # Skip frames to improve performance
        if self.frame_count % self.frame_skip != 0:
            # Return cached results
            return self.last_faces
        
        try:
            # Downscale frame for faster detection
            if self.detection_scale < 1.0:
                small_frame = cv2.resize(frame, None, fx=self.detection_scale, fy=self.detection_scale)
            else:
                small_frame = frame
            
            # Run detection on smaller frame
            if self.detection_method == "dnn":
                faces = self._detect_dnn(small_frame)
            else:
                faces = self._detect_haar(small_frame)
            
            # Scale bounding boxes back to original size
            if self.detection_scale < 1.0:
                scale_factor = 1.0 / self.detection_scale
                faces = [(int(x * scale_factor), int(y * scale_factor), 
                         int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in faces]
            
            # Cache results
            self.last_faces = faces
            return faces
        except Exception as e:
            print(f"Detection error: {e}, falling back to Haar Cascade")
            # Fall back to Haar Cascade on error
            if self.detection_method == "dnn":
                try:
                    self._init_haar_detector()
                    self.detection_method = "haar"
                except:
                    pass
            return self.last_faces
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN (optimized for speed)."""
        h, w = frame.shape[:2]
        
        # Validate frame dimensions
        if h <= 0 or w <= 0:
            return []
        
        # Use smaller input size for faster inference
        input_size = getattr(self, 'dnn_input_size', 300)
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (input_size, input_size)),
            1.0,
            (input_size, input_size),
            (104.0, 177.0, 123.0)
        )
        
        # Run inference
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Clamp to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Convert to (x, y, width, height)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade (CPU)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
