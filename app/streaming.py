"""Video streaming service that manages capture and broadcasting."""
import asyncio
import logging
import time
import psutil
from typing import Optional, List

from camera import CameraCapture
from encoder import FrameEncoder
from client_manager import ClientManager
from yolo_detector import YOLOv5Detector
from system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class StreamingService:
    """Manages video capture, encoding, and client broadcasting."""
    
    def __init__(
        self,
        camera: Optional[CameraCapture] = None,
        encoder: Optional[FrameEncoder] = None,
        client_manager: Optional[ClientManager] = None,
        detector: Optional[YOLOv5Detector] = None
    ):
        """Initialize streaming service.
        
        Args:
            camera: Camera capture instance (creates default if None)
            encoder: Frame encoder instance (creates default if None)
            client_manager: Client manager instance (creates default if None)
            detector: YOLO detector instance (creates default if None)
        """
        self.camera = camera or CameraCapture()
        self.encoder = encoder or FrameEncoder()
        self.client_manager = client_manager or ClientManager()
        self.detector = detector or YOLOv5Detector()
        self.system_monitor = SystemMonitor()
        
        self.latest_frame: Optional[bytes] = None
        self.is_running = False
        self.capture_task: Optional[asyncio.Task] = None
        
        # Feature flags
        self._detection_enabled = True  # Enabled by default
        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self.fps = 0.0
        self._fps_update_interval = 1.0  # Update FPS every second
        
        # Memory tracking
        self.process = psutil.Process()
    
    async def start(self) -> None:
        """Start the streaming service."""
        if self.is_running:
            logger.warning("Streaming service is already running")
            return
        
        self.is_running = True
        # Python 3.6 compatible: use ensure_future instead of create_task
        self.capture_task = asyncio.ensure_future(self._capture_loop())
        logger.info("Streaming service started")
    
    async def stop(self) -> None:
        """Stop the streaming service."""
        self.is_running = False
        
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
        
        self.camera.release()
        logger.info("Streaming service stopped")
    
    async def _capture_loop(self) -> None:
        """Main capture loop that reads frames and broadcasts to clients."""
        logger.info("=== CAPTURE LOOP STARTED ===")
        logger.info("Camera is_opened: {}".format(self.camera.is_opened()))
        logger.info("Is running: {}".format(self.is_running))
        
        frame_count = 0
        failed_reads = 0
        try:
            while self.is_running:
                # Read frame from camera
                success, frame = self.camera.read()
                if not success:
                    failed_reads += 1
                    if frame_count == 0:
                        logger.error("Failed to read first frame from camera (attempt {})".format(failed_reads))
                    elif failed_reads % 100 == 0:
                        logger.warning("Camera read failures: {}".format(failed_reads))
                    await asyncio.sleep(0.01)
                    continue
                
                if failed_reads > 0:
                    logger.info("Camera read recovered after {} failures".format(failed_reads))
                    failed_reads = 0
                
                if frame_count == 0:
                    logger.info("=== FIRST FRAME READ: {}x{} ===".format(frame.shape[1], frame.shape[0]))
                
                frame_count += 1
                
                # Apply object detection if enabled
                if self._detection_enabled:
                    detections = self.detector.detect(frame)
                    frame = self.detector.draw_detections(frame, detections)
                
                # Encode frame to JPEG
                success, jpeg_bytes = self.encoder.encode(frame)
                if not success:
                    logger.warning("Failed to encode frame {}".format(frame_count))
                    await asyncio.sleep(0)
                    continue
                
                if frame_count == 1:
                    logger.info("=== FIRST FRAME ENCODED: {} bytes ===".format(len(jpeg_bytes)))
                elif frame_count % 100 == 0:
                    logger.info("Frame {} encoded: {} bytes".format(frame_count, len(jpeg_bytes)))
                
                # Store latest frame
                self.latest_frame = jpeg_bytes
                
                # Update FPS counter
                self._frame_count += 1
                current_time = time.time()
                elapsed = current_time - self._fps_start_time
                if elapsed >= self._fps_update_interval:
                    self.fps = self._frame_count / elapsed
                    if self._frame_count > 0:
                        logger.info("FPS: {:.1f}, Frames: {}, Clients: {}".format(self.fps, frame_count, self.client_manager.get_client_count()))
                    self._frame_count = 0
                    self._fps_start_time = current_time
                
                # Broadcast to all clients
                client_count = self.client_manager.get_client_count()
                if client_count > 0:
                    await self.client_manager.broadcast(jpeg_bytes)
                    if frame_count == 1:
                        logger.info("=== FIRST FRAME BROADCAST to {} clients ===".format(client_count))
                elif frame_count % 100 == 0:
                    logger.warning("Frame {} produced but NO CLIENTS connected".format(frame_count))
                
                # Yield to event loop
                await asyncio.sleep(0)
        
        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            self.camera.release()
    
    def get_latest_frame(self) -> Optional[bytes]:
        """Get the most recent encoded frame.
        
        Returns:
            Latest JPEG frame bytes, or None if no frame available
        """
        return self.latest_frame
    
    def get_client_count(self) -> int:
        """Get the number of connected clients."""
        return self.client_manager.get_client_count()
    
    def get_fps(self) -> float:
        """Get the current frames per second.
        
        Returns:
            Current FPS
        """
        return round(self.fps, 1)
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB and percentage
        """
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            return {
                "used_mb": round(memory_mb, 1),
                "percent": round(memory_percent, 1)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"used_mb": 0, "percent": 0}
    
    def enable_detection(self, enabled: bool) -> None:
        """Enable or disable detection.
        
        Args:
            enabled: Whether to enable detection
        """
        self._detection_enabled = enabled
        logger.info("Detection {}".format("enabled" if enabled else "disabled"))
    
    def is_detection_enabled(self) -> bool:
        """Check if detection is enabled.
        
        Returns:
            True if detection is enabled
        """
        return self._detection_enabled
    
    def set_selected_classes(self, class_indices: List[int]) -> None:
        """Set which object classes to detect.
        
        Args:
            class_indices: List of class indices to detect
        """
        self.detector.set_selected_classes(class_indices)
    
    def get_selected_classes(self) -> List[int]:
        """Get currently selected class indices.
        
        Returns:
            List of selected class indices
        """
        return self.detector.get_selected_classes()

