"""Unit tests for client manager."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from app.client_manager import ClientManager


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.sent_data = []
        self.closed = False
    
    async def send_bytes(self, data):
        """Mock send_bytes method."""
        if self.should_fail:
            raise RuntimeError("Send failed")
        self.sent_data.append(data)
    
    async def close(self):
        """Mock close method."""
        self.closed = True


@pytest.mark.asyncio
class TestClientManager:
    """Test ClientManager class."""
    
    async def test_init(self):
        """Test client manager initialization."""
        manager = ClientManager()
        assert len(manager.clients) == 0
    
    async def test_add_client(self):
        """Test adding a client."""
        manager = ClientManager()
        ws = MockWebSocket()
        
        await manager.add_client(ws)
        assert len(manager.clients) == 1
        assert ws in manager.clients
    
    async def test_remove_client(self):
        """Test removing a client."""
        manager = ClientManager()
        ws = MockWebSocket()
        
        await manager.add_client(ws)
        await manager.remove_client(ws)
        
        assert len(manager.clients) == 0
        assert ws not in manager.clients
    
    async def test_remove_nonexistent_client(self):
        """Test removing a client that doesn't exist."""
        manager = ClientManager()
        ws = MockWebSocket()
        
        # Should not raise exception
        await manager.remove_client(ws)
        assert len(manager.clients) == 0
    
    async def test_broadcast_to_single_client(self):
        """Test broadcasting to a single client."""
        manager = ClientManager()
        ws = MockWebSocket()
        test_data = b"test frame data"
        
        await manager.add_client(ws)
        await manager.broadcast(test_data)
        
        assert len(ws.sent_data) == 1
        assert ws.sent_data[0] == test_data
    
    async def test_broadcast_to_multiple_clients(self):
        """Test broadcasting to multiple clients."""
        manager = ClientManager()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        test_data = b"test frame data"
        
        await manager.add_client(ws1)
        await manager.add_client(ws2)
        await manager.add_client(ws3)
        
        await manager.broadcast(test_data)
        
        assert len(ws1.sent_data) == 1
        assert len(ws2.sent_data) == 1
        assert len(ws3.sent_data) == 1
    
    async def test_broadcast_removes_failed_clients(self):
        """Test that failed clients are removed during broadcast."""
        manager = ClientManager()
        ws_good = MockWebSocket(should_fail=False)
        ws_bad = MockWebSocket(should_fail=True)
        test_data = b"test frame data"
        
        await manager.add_client(ws_good)
        await manager.add_client(ws_bad)
        
        assert len(manager.clients) == 2
        
        await manager.broadcast(test_data)
        
        # Good client should remain, bad client should be removed
        assert len(manager.clients) == 1
        assert ws_good in manager.clients
        assert ws_bad not in manager.clients
    
    async def test_broadcast_to_no_clients(self):
        """Test broadcasting with no connected clients."""
        manager = ClientManager()
        test_data = b"test frame data"
        
        # Should not raise exception
        await manager.broadcast(test_data)
    
    async def test_get_client_count(self):
        """Test getting client count."""
        manager = ClientManager()
        assert manager.get_client_count() == 0
        
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.add_client(ws1)
        assert manager.get_client_count() == 1
        
        await manager.add_client(ws2)
        assert manager.get_client_count() == 2
        
        await manager.remove_client(ws1)
        assert manager.get_client_count() == 1
    
    async def test_concurrent_client_operations(self):
        """Test concurrent client additions and removals."""
        manager = ClientManager()
        clients = [MockWebSocket() for _ in range(10)]
        
        # Add all clients concurrently
        await asyncio.gather(*[manager.add_client(ws) for ws in clients])
        assert manager.get_client_count() == 10
        
        # Remove half concurrently
        await asyncio.gather(*[manager.remove_client(ws) for ws in clients[:5]])
        assert manager.get_client_count() == 5
    
    async def test_broadcast_with_varying_client_count(self):
        """Test broadcasting while clients connect/disconnect."""
        manager = ClientManager()
        test_data = b"test frame"
        
        # Start with 3 clients
        clients = [MockWebSocket() for _ in range(3)]
        for ws in clients:
            await manager.add_client(ws)
        
        # Broadcast
        await manager.broadcast(test_data)
        assert all(len(ws.sent_data) == 1 for ws in clients)
        
        # Add 2 more clients
        new_clients = [MockWebSocket() for _ in range(2)]
        for ws in new_clients:
            await manager.add_client(ws)
        
        # Broadcast again
        await manager.broadcast(test_data)
        assert all(len(ws.sent_data) == 2 for ws in clients)
        assert all(len(ws.sent_data) == 1 for ws in new_clients)
        
        # Remove 2 original clients
        await manager.remove_client(clients[0])
        await manager.remove_client(clients[1])
        
        # Broadcast with remaining clients
        await manager.broadcast(test_data)
        assert manager.get_client_count() == 3
