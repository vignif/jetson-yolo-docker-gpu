"""YOLOv8n object detector with class selection."""
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


class YOLOv8Detector:
    """YOLOv8n detector optimized for Jetson Nano."""
    
    def __init__(self, conf_threshold=0.4, iou_threshold=0.45):
        """Initialize YOLOv8n detector.
        
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
        
        logger.info("Loading YOLOv8n (nano - lightweight)...")
        try:
            # Use ultralytics YOLOv8n
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # nano model
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("YOLOv8n initialized on {}".format(self.device))
            logger.info("Detecting {} object classes".format(len(COCO_CLASSES)))
            logger.info("Backend: YOLOv8n-GPU" if torch.cuda.is_available() else "Backend: YOLOv8n-CPU")
        except Exception as e:
            logger.error("Failed to load YOLOv8n: {}".format(e))
            raise
    
    def set_selected_classes(self, class_indices: List[int]) -> None:
        """Set which classes to detect.
        
        Args:
            class_indices: List of class indices to detect
        """
        self.selected_classes = set(class_indices)
        class_names = [COCO_CLASSES[i] for i in class_indices if i < len(COCO_CLASSES)]
        logger.info("Selected {} classes: {}".format(len(class_names), ', '.join(class_names[:10]) + ('...' if len(class_names) > 10 else '')))
    
    def get_selected_classes(self) -> List[int]:
        """Get currently selected class indices.
        
        Returns:
            List of selected class indices
        """
        return sorted(list(self.selected_classes))
    
    def set_conf_threshold(self, threshold: float) -> None:
        """Set confidence threshold.
        
        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        self.conf_threshold = max(0.0, min(1.0, threshold))
        logger.info("Confidence threshold set to {:.2f}".format(self.conf_threshold))
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect objects in frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections, each as dict with keys: bbox, confidence, class_id, class_name
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
            
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    # Filter by selected classes
                    if class_id not in self.selected_classes:
                        continue
                    
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'
                    })
            
            return detections
        
        except Exception as e:
            logger.error("Detection error: {}".format(e))
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes on frame.
        
        Args:
            frame: Input image
            detections: List of detections from detect()
            
        Returns:
            Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = "{}: {:.2f}".format(class_name, confidence)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )
            
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return frame
    
    def get_backend_name(self) -> str:
        """Get the detector backend name.
        
        Returns:
            Backend name string
        """
        return "YOLOv8n-GPU" if torch.cuda.is_available() else "YOLOv8n-CPU"
    
    @staticmethod
    def get_available_classes() -> List[dict]:
        """Get list of available detection classes.
        
        Returns:
            List of dicts with 'id' and 'name' keys
        """
        return [{'id': i, 'name': name} for i, name in enumerate(COCO_CLASSES)]
