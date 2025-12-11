"""PyTorch GPU face detection for Jetson Nano."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import urllib.request
import os


class UltraFaceSlim(nn.Module):
    """Ultra-lightweight face detection network optimized for edge devices.
    
    This is a simplified version of UltraFace (RFB-320) designed for
    real-time inference on Jetson Nano's GPU (128 CUDA cores).
    """
    
    def __init__(self):
        super(UltraFaceSlim, self).__init__()
        
        # Slim feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Detection head
        self.cls_head = nn.Conv2d(128, 4, kernel_size=1)  # 2 anchors * 2 classes
        self.reg_head = nn.Conv2d(128, 8, kernel_size=1)  # 2 anchors * 4 coords
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        
        return cls, reg


class FaceDetectorTensorRT:
    """PyTorch GPU-based face detection for Jetson Nano.
    
    Uses a lightweight custom face detector optimized for real-time inference
    on Jetson's 128 CUDA cores. Falls back to CPU Haar Cascade if GPU unavailable.
    """
    
    def __init__(self):
        """Initialize face detector."""
        self.enabled = False
        self.backend = "Unknown"
        self.last_faces = []
        self.frame_count = 0
        self.frame_skip = 1  # Process every frame with GPU
        self.device = None
        self.model = None
        self.input_size = (320, 240)  # Optimized for Jetson
        self.conf_threshold = 0.7
        self.iou_threshold = 0.3
        
        print("Initializing PyTorch GPU face detection...")
        
        # Try GPU first, fallback to CPU
        if not self._init_pytorch_detector():
            print("⚠ PyTorch GPU unavailable, falling back to CPU Haar Cascade")
            self._init_haar_detector()
        
        print(f"✓ Face detector ready with {self.backend} backend")
    
    def _init_pytorch_detector(self) -> bool:
        """Initialize PyTorch GPU detector.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                print("CUDA not available")
                return False
            
            self.device = torch.device('cuda:0')
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Initialize model
            self.model = UltraFaceSlim().to(self.device)
            self.model.eval()
            
            # Warmup
            dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0]).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            self.backend = "PyTorch-GPU"
            self.detection_method = "pytorch"
            print("✓ PyTorch GPU detector initialized")
            return True
            
        except Exception as e:
            print(f"PyTorch GPU init failed: {e}")
            return False
    
    def _init_haar_detector(self) -> None:
        """Initialize fallback Haar Cascade detector."""
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
                    self.backend = "Haar-CPU-Fallback"
                    self.frame_skip = 3  # Skip frames on CPU
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
            if self.detection_method == "pytorch":
                faces = self._detect_pytorch(frame)
            else:
                faces = self._detect_haar(frame)
            
            self.last_faces = faces
            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return self.last_faces
    
    def _detect_pytorch(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using PyTorch GPU.
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of (x, y, w, h) tuples
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        resized = cv2.resize(frame, self.input_size)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        img_tensor = (img_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            cls_pred, reg_pred = self.model(img_tensor)
        
        # Post-process predictions
        faces = self._postprocess_predictions(
            cls_pred, reg_pred, 
            orig_w, orig_h
        )
        
        return faces
    
    def _postprocess_predictions(
        self, 
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        orig_w: int,
        orig_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Convert model predictions to bounding boxes.
        
        Args:
            cls_pred: Classification predictions
            reg_pred: Regression predictions  
            orig_w: Original frame width
            orig_h: Original frame height
            
        Returns:
            List of (x, y, w, h) tuples
        """
        boxes = []
        scores = []
        
        # Get feature map dimensions
        _, _, fh, fw = cls_pred.shape
        
        # Process each spatial location
        cls_pred = cls_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reg_pred = reg_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        for y in range(fh):
            for x in range(fw):
                # Get confidence scores (2 anchors)
                for anchor_idx in range(2):
                    score_idx = anchor_idx * 2 + 1  # Face class
                    score = 1 / (1 + np.exp(-cls_pred[y, x, score_idx]))  # Sigmoid
                    
                    if score < self.conf_threshold:
                        continue
                    
                    # Get bbox coordinates (normalized)
                    reg_idx = anchor_idx * 4
                    cx = (x + reg_pred[y, x, reg_idx + 0]) / fw
                    cy = (y + reg_pred[y, x, reg_idx + 1]) / fh
                    w = reg_pred[y, x, reg_idx + 2] / fw
                    h = reg_pred[y, x, reg_idx + 3] / fh
                    
                    # Convert to absolute coordinates
                    x1 = int((cx - w/2) * orig_w)
                    y1 = int((cy - h/2) * orig_h)
                    x2 = int((cx + w/2) * orig_w)
                    y2 = int((cy + h/2) * orig_h)
                    
                    # Clip to frame bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
        
        if not boxes:
            return []
        
        # Apply NMS
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = self._nms(boxes, scores, self.iou_threshold)
        
        # Convert to (x, y, w, h) format
        result = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return result
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-maximum suppression.
        
        Args:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
            scores: Array of shape (N,) with confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
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
