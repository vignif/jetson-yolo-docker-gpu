"""WebSocket client manager for broadcasting video frames."""
from typing import Set
from fastapi import WebSocket
import asyncio
import logging

logger = logging.getLogger(__name__)


class ClientManager:
    """Manages WebSocket clients and broadcasts frames."""
    
    def __init__(self, send_timeout: float = 0.02):
        """Initialize client manager.
        
        Args:
            send_timeout: Maximum time to wait when sending to a client (seconds)
        """
        self.clients: Set[WebSocket] = set()
        self.send_timeout = send_timeout
    
    def add_client(self, websocket: WebSocket) -> None:
        """Add a new WebSocket client."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    def remove_client(self, websocket: WebSocket) -> None:
        """Remove a WebSocket client."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    def get_client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.clients)
    
    async def broadcast(self, data: bytes) -> None:
        """Broadcast data to all connected clients.
        
        Automatically removes slow or disconnected clients.
        
        Args:
            data: Binary data to send (e.g., JPEG frame)
        """
        if not self.clients:
            return
        
        send_tasks = []
        for ws in list(self.clients):
            try:
                # Check if connection is still valid
                if ws.application_state.name != 'CONNECTED':
                    self.clients.discard(ws)
                    continue
                
                # Create send task with timeout to avoid blocking
                send_tasks.append(
                    asyncio.wait_for(
                        ws.send_bytes(data),
                        timeout=self.send_timeout
                    )
                )
            except Exception as e:
                logger.debug(f"Error preparing send task: {e}")
                self.clients.discard(ws)
        
        if send_tasks:
            # Execute all sends concurrently, ignoring exceptions
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Clean up clients that failed to send
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.debug(f"Send failed for client: {result}")
