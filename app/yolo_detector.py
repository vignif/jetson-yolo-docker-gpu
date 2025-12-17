"""YOLOv5n object detector with class selection - Python 3.6 compatible."""
import logging
import torch
import cv2
import numpy as np
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)


# COCO dataset classes (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLOv5Detector:
    """YOLOv5n detector optimized for Jetson Nano with Python 3.6."""
    
    def __init__(self, conf_threshold=0.4, iou_threshold=0.45):
        """Initialize YOLOv5n detector.
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Selected classes to detect (all by default)
        self.selected_classes: Set[int] = set(range(len(COCO_CLASSES)))
        
        logger.info("Device: {}".format(self.device))
        if torch.cuda.is_available():
            logger.info("CUDA device: {}".format(torch.cuda.get_device_name(0)))
            logger.info("CUDA memory: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
        
        logger.info("Loading YOLOv5n (nano - lightweight) via torch.hub...")
        try:
            # Load YOLOv5n from v6.0 tag (last version without ultralytics dependency)
            # force_reload=True clears the cached version that requires ultralytics
            self.model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5n', pretrained=True, force_reload=True)
            self.model.to(self.device)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            
            backend = "YOLOv5n-GPU" if self.device.type == 'cuda' else "YOLOv5n-CPU"
            logger.info("{} initialized on {}".format(backend, self.device))
            logger.info("Detecting {} object classes".format(len(COCO_CLASSES)))
        except Exception as e:
            logger.error("Failed to load YOLOv5n: {}".format(e))
            raise
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        """Detect objects in frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of (x1, y1, x2, y2, class_id, confidence) tuples
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # YOLOv5 expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            # Extract detections
            detections = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                class_id = int(cls)
                
                # Filter by selected classes
                if class_id not in self.selected_classes:
                    continue
                
                # Filter by confidence
                if conf < self.conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2, y2, class_id, float(conf)))
            
            return detections
        except Exception as e:
            logger.error("Detection error: {}".format(e))
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, int, float]]) -> np.ndarray:
        """Draw bounding boxes and labels on frame.
        
        Args:
            frame: BGR image from OpenCV
            detections: List of (x1, y1, x2, y2, class_id, confidence) tuples
            
        Returns:
            Frame with drawn detections
        """
        for x1, y1, x2, y2, class_id, conf in detections:
            # Get class name
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
            
            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = "{}: {:.2f}".format(class_name, conf)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text (black on green)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def set_selected_classes(self, class_indices: List[int]) -> None:
        """Set which object classes to detect.
        
        Args:
            class_indices: List of class indices (0-79) to detect
        """
        self.selected_classes = set(class_indices)
        logger.info("Selected {} classes for detection".format(len(self.selected_classes)))
    
    def get_selected_classes(self) -> List[int]:
        """Get currently selected object classes.
        
        Returns:
            List of selected class indices
        """
        return sorted(list(self.selected_classes))
    
    def update_params(self, conf_threshold: float = None, iou_threshold: float = None) -> None:
        """Update detection parameters.
        
        Args:
            conf_threshold: New confidence threshold
            iou_threshold: New IOU threshold for NMS
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            self.model.conf = conf_threshold
            logger.info("Updated conf_threshold to {:.2f}".format(conf_threshold))
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            self.model.iou = iou_threshold
            logger.info("Updated iou_threshold to {:.2f}".format(iou_threshold))
    
    @staticmethod
    def get_available_classes() -> List[dict]:
        """Get list of all available COCO classes.
        
        Returns:
            List of {"id": int, "name": str} dicts
        """
        return [{"id": i, "name": name} for i, name in enumerate(COCO_CLASSES)]
    
    def get_backend_name(self) -> str:
        """Get detector backend name.
        
        Returns:
            Backend name string
        """
        return "YOLOv5n-GPU" if self.device.type == 'cuda' else "YOLOv5n-CPU"
