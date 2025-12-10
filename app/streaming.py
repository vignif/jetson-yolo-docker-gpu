"""Video streaming service that manages capture and broadcasting."""
import asyncio
import logging
from typing import Optional

from camera import CameraCapture
from encoder import FrameEncoder
from client_manager import ClientManager

logger = logging.getLogger(__name__)


class StreamingService:
    """Manages video capture, encoding, and client broadcasting."""
    
    def __init__(
        self,
        camera: Optional[CameraCapture] = None,
        encoder: Optional[FrameEncoder] = None,
        client_manager: Optional[ClientManager] = None
    ):
        """Initialize streaming service.
        
        Args:
            camera: Camera capture instance (creates default if None)
            encoder: Frame encoder instance (creates default if None)
            client_manager: Client manager instance (creates default if None)
        """
        self.camera = camera or CameraCapture()
        self.encoder = encoder or FrameEncoder()
        self.client_manager = client_manager or ClientManager()
        
        self.latest_frame: Optional[bytes] = None
        self.is_running = False
        self.capture_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the streaming service."""
        if self.is_running:
            logger.warning("Streaming service is already running")
            return
        
        self.is_running = True
        self.capture_task = asyncio.create_task(self._capture_loop())
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
        try:
            while self.is_running:
                # Read frame from camera
                success, frame = self.camera.read()
                if not success:
                    await asyncio.sleep(0.01)
                    continue
                
                # Encode frame to JPEG
                success, jpeg_bytes = self.encoder.encode(frame)
                if not success:
                    await asyncio.sleep(0)
                    continue
                
                # Store latest frame
                self.latest_frame = jpeg_bytes
                
                # Broadcast to all clients
                await self.client_manager.broadcast(jpeg_bytes)
                
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
