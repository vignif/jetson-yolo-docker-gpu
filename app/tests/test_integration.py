"""Integration tests with multiple clients."""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from app.streaming import StreamingService
from app.client_manager import ClientManager


class MockWebSocket:
    """Mock WebSocket for integration testing."""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.frames_received = 0
        self.total_bytes = 0
        self.closed = False
    
    async def send_bytes(self, data):
        """Mock send_bytes method."""
        self.frames_received += 1
        self.total_bytes += len(data)
    
    async def close(self):
        """Mock close method."""
        self.closed = True


@pytest.mark.asyncio
class TestMultiClientIntegration:
    """Integration tests with varying client counts."""
    
    async def test_single_client_streaming(self):
        """Test streaming to a single client."""
        manager = ClientManager()
        ws = MockWebSocket(1)
        
        await manager.add_client(ws)
        
        # Simulate 10 frame broadcasts
        for i in range(10):
            test_data = b"frame_" + str(i).encode()
            await manager.broadcast(test_data)
        
        assert ws.frames_received == 10
        assert manager.get_client_count() == 1
    
    async def test_multiple_clients_streaming(self):
        """Test streaming to multiple clients simultaneously."""
        manager = ClientManager()
        clients = [MockWebSocket(i) for i in range(5)]
        
        # Add all clients
        for ws in clients:
            await manager.add_client(ws)
        
        assert manager.get_client_count() == 5
        
        # Broadcast 20 frames
        for i in range(20):
            test_data = b"frame_" + str(i).encode()
            await manager.broadcast(test_data)
        
        # All clients should receive all frames
        for ws in clients:
            assert ws.frames_received == 20
        
        assert manager.get_client_count() == 5
    
    async def test_clients_joining_during_streaming(self):
        """Test clients joining while streaming is active."""
        manager = ClientManager()
        
        # Start with 2 clients
        initial_clients = [MockWebSocket(i) for i in range(2)]
        for ws in initial_clients:
            await manager.add_client(ws)
        
        # Broadcast 10 frames
        for i in range(10):
            await manager.broadcast(b"frame_" + str(i).encode())
        
        # Add 3 more clients
        new_clients = [MockWebSocket(i + 2) for i in range(3)]
        for ws in new_clients:
            await manager.add_client(ws)
        
        assert manager.get_client_count() == 5
        
        # Broadcast 10 more frames
        for i in range(10, 20):
            await manager.broadcast(b"frame_" + str(i).encode())
        
        # Initial clients should have all 20 frames
        for ws in initial_clients:
            assert ws.frames_received == 20
        
        # New clients should have only 10 frames
        for ws in new_clients:
            assert ws.frames_received == 10
    
    async def test_clients_leaving_during_streaming(self):
        """Test clients leaving while streaming is active."""
        manager = ClientManager()
        
        # Start with 5 clients
        clients = [MockWebSocket(i) for i in range(5)]
        for ws in clients:
            await manager.add_client(ws)
        
        # Broadcast 10 frames
        for i in range(10):
            await manager.broadcast(b"frame_" + str(i).encode())
        
        # Remove 2 clients
        await manager.remove_client(clients[0])
        await manager.remove_client(clients[1])
        
        assert manager.get_client_count() == 3
        
        # Broadcast 10 more frames
        for i in range(10, 20):
            await manager.broadcast(b"frame_" + str(i).encode())
        
        # Removed clients should have only first 10 frames
        assert clients[0].frames_received == 10
        assert clients[1].frames_received == 10
        
        # Remaining clients should have all 20 frames
        for ws in clients[2:]:
            assert ws.frames_received == 20
    
    async def test_high_client_churn(self):
        """Test rapid client connects and disconnects."""
        manager = ClientManager()
        
        # Simulate 50 clients connecting and disconnecting
        for i in range(50):
            ws = MockWebSocket(i)
            await manager.add_client(ws)
            
            # Broadcast a frame
            await manager.broadcast(b"frame_" + str(i).encode())
            
            # Remove every other client
            if i % 2 == 0:
                await manager.remove_client(ws)
        
        # Should have ~25 clients remaining
        assert manager.get_client_count() == 25
    
    async def test_zero_to_many_clients(self):
        """Test scaling from zero to many clients."""
        manager = ClientManager()
        
        # Start with no clients
        assert manager.get_client_count() == 0
        
        # Broadcast should not fail
        await manager.broadcast(b"test_frame")
        
        # Add clients progressively
        clients = []
        for i in range(1, 11):
            ws = MockWebSocket(i)
            await manager.add_client(ws)
            clients.append(ws)
            
            assert manager.get_client_count() == i
            
            # Broadcast after each addition
            await manager.broadcast(b"frame_" + str(i).encode())
        
        # Each client should have received frames from when they joined
        for idx, ws in enumerate(clients):
            expected_frames = 10 - idx
            assert ws.frames_received == expected_frames
    
    async def test_many_to_zero_clients(self):
        """Test scaling down from many to zero clients."""
        manager = ClientManager()
        
        # Start with 10 clients
        clients = [MockWebSocket(i) for i in range(10)]
        for ws in clients:
            await manager.add_client(ws)
        
        assert manager.get_client_count() == 10
        
        # Remove clients one by one
        for idx, ws in enumerate(clients):
            await manager.remove_client(ws)
            assert manager.get_client_count() == 10 - idx - 1
            
            # Broadcast should still work
            await manager.broadcast(b"frame_after_remove")
        
        assert manager.get_client_count() == 0
    
    async def test_concurrent_client_operations_under_load(self):
        """Test concurrent operations with simulated load."""
        manager = ClientManager()
        
        async def add_and_broadcast(client_id):
            ws = MockWebSocket(client_id)
            await manager.add_client(ws)
            await asyncio.sleep(0.01)  # Simulate work
            for _ in range(5):
                await manager.broadcast(b"frame_data")
            await manager.remove_client(ws)
            return ws.frames_received
        
        # Run 20 concurrent client sessions
        results = await asyncio.gather(*[add_and_broadcast(i) for i in range(20)])
        
        # All clients should have received some frames
        assert all(frames > 0 for frames in results)
        
        # All clients should be removed
        assert manager.get_client_count() == 0
