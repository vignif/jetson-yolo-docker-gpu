"""Pytest configuration and fixtures."""
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_frame():
    """Provide a mock camera frame."""
    import numpy as np
    return np.zeros((720, 1280, 3), dtype=np.uint8)
