"""Integration tests for streaming pipeline."""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from app.streaming import StreamingService


@pytest.mark.asyncio
class TestStreamingService:
    """Test StreamingService class."""
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_init(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test streaming service initialization."""
        service = StreamingService()
        
        assert service.camera is not None
        assert service.face_detector is not None
        assert service.object_detector is not None
        assert service.client_manager is not None
        assert service.running is False
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_start_stop(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test starting and stopping the service."""
        service = StreamingService()
        
        # Mock camera to return frames
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cam.return_value.read.return_value = (True, test_frame)
        mock_cam.return_value.is_opened.return_value = True
        
        # Mock face detector
        mock_fd.return_value.detect.return_value = []
        
        # Mock object detector
        mock_od.return_value.detect.return_value = []
        
        # Mock client manager broadcast
        mock_cm.return_value.broadcast = AsyncMock()
        mock_cm.return_value.get_client_count.return_value = 0
        
        # Start service
        service.start()
        assert service.running is True
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop service
        await service.stop()
        assert service.running is False
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_frame_encoding(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test frame encoding to JPEG."""
        service = StreamingService()
        
        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Mock detectors
        mock_fd.return_value.detect.return_value = []
        mock_od.return_value.detect.return_value = []
        
        # Test encoding
        encoded = service._encode_frame(test_frame)
        
        assert encoded is not None
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_detection_toggles(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test enabling/disabling detections."""
        service = StreamingService()
        
        # Test face detection toggle
        assert service.face_detection_enabled is True
        service.toggle_face_detection(False)
        assert service.face_detection_enabled is False
        service.toggle_face_detection(True)
        assert service.face_detection_enabled is True
        
        # Test object detection toggle
        assert service.object_detection_enabled is False
        service.toggle_object_detection(True)
        assert service.object_detection_enabled is True
        service.toggle_object_detection(False)
        assert service.object_detection_enabled is False
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_camera_read_failure_handling(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test handling of camera read failures."""
        service = StreamingService()
        
        # Mock camera to fail reading
        mock_cam.return_value.read.return_value = (False, None)
        mock_cam.return_value.is_opened.return_value = True
        
        service.start()
        await asyncio.sleep(0.1)
        
        # Service should still be running, waiting for camera
        assert service.running is True
        
        await service.stop()
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_stats_tracking(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test statistics tracking."""
        service = StreamingService()
        
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cam.return_value.read.return_value = (True, test_frame)
        mock_cam.return_value.is_opened.return_value = True
        mock_fd.return_value.detect.return_value = []
        mock_od.return_value.detect.return_value = []
        mock_cm.return_value.broadcast = AsyncMock()
        mock_cm.return_value.get_client_count.return_value = 1
        
        service.start()
        await asyncio.sleep(0.5)
        
        stats = service.get_stats()
        
        assert 'fps' in stats
        assert 'clients' in stats
        assert 'face_detection_enabled' in stats
        assert 'object_detection_enabled' in stats
        assert stats['fps'] >= 0
        
        await service.stop()
    
    @patch('app.streaming.CameraCapture')
    @patch('app.streaming.FaceDetector')
    @patch('app.streaming.ObjectDetector')
    @patch('app.streaming.ClientManager')
    async def test_multiple_restarts(self, mock_cm, mock_od, mock_fd, mock_cam):
        """Test restarting the service multiple times."""
        service = StreamingService()
        
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cam.return_value.read.return_value = (True, test_frame)
        mock_cam.return_value.is_opened.return_value = True
        mock_fd.return_value.detect.return_value = []
        mock_od.return_value.detect.return_value = []
        mock_cm.return_value.broadcast = AsyncMock()
        mock_cm.return_value.get_client_count.return_value = 0
        
        for _ in range(3):
            service.start()
            assert service.running is True
            await asyncio.sleep(0.1)
            await service.stop()
            assert service.running is False
