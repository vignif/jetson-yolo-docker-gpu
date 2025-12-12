"""Simple object detection using torchvision MobileNet SSD for Jetson Nano."""
import cv2
import numpy as np
import torch
import torchvision
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Simple object detection using MobileNetV2 SSD optimized for Jetson Nano.
    
    Uses torchvision's pre-trained model for real-time detection on 128 CUDA cores.
    """
    
    def __init__(self, conf_threshold: float = 0.5):
        """Initialize object detector.
        
        Args:
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = None
        self.model = None
        self.class_names = self._get_coco_names()
        
        self._init_detector()
    
    def _get_coco_names(self) -> List[str]:
        """Get COCO dataset class names."""
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def _init_detector(self) -> None:
        """Initialize detection model on GPU if available."""
        try:
            # Check CUDA availability
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("Device: {}".format(self.device))
            
            if torch.cuda.is_available():
                logger.info("CUDA device: {}".format(torch.cuda.get_device_name(0)))
                logger.info("CUDA memory: {:.2f} GB".format(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                ))
            
            # Load pre-trained SSD model from torchvision
            logger.info("Loading SSD MobileNetV2 model...")
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Object detector initialized on {}".format(self.device))
            logger.info("Detecting {} object classes".format(len(self.class_names) - 1))
            
        except Exception as e:
            logger.error("Failed to initialize detector: {}".format(e))
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """Detect objects in frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of (x, y, w, h, class_name, confidence) tuples
        """
        if self.model is None:
            return []
        
        try:
            # Prepare input
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Parse results
            detections = []
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score >= self.conf_threshold and label < len(self.class_names):
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    class_name = self.class_names[label]
                    detections.append((x1, y1, w, h, class_name, float(score)))
            
            return detections
            
        except Exception as e:
            logger.error("Detection error: {}".format(e))
            return []
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Tuple[int, int, int, int, str, float]]) -> np.ndarray:
        """Draw detection boxes and labels on frame.
        
        Args:
            frame: BGR image
            detections: List of (x, y, w, h, class_name, confidence)
            
        Returns:
            Frame with drawn detections
        """
        for x, y, w, h, class_name, conf in detections:
            # Skip background class
            if class_name == '__background__':
                continue
                
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = "{}: {:.2f}".format(class_name, conf)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label background
            cv2.rectangle(frame, 
                         (x, y - label_size[1] - 5),
                         (x + label_size[0], y),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def get_backend_name(self) -> str:
        """Get the name of the detection backend."""
        if self.model is None:
            return "None"
        return "SSD-MobileNet-GPU" if self.device.type == "cuda" else "SSD-MobileNet-CPU"
        if self.model is None:
            return "None"
        return "YOLOv5n-GPU" if self.device.type == 'cuda' else "YOLOv5n-CPU"
