"""Unit tests for camera module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.camera import CameraCapture


class TestCameraCapture:
    """Test CameraCapture class."""
    
    def test_init(self):
        """Test camera initialization."""
        camera = CameraCapture(width=1280, height=720, fps=30)
        assert camera.width == 1280
        assert camera.height == 720
        assert camera.fps == 30
        assert camera.cap is None
        assert not camera._initialized
    
    def test_pipeline_generation(self):
        """Test GStreamer pipeline string generation."""
        camera = CameraCapture(width=1920, height=1080, fps=60)
        pipeline = camera._build_pipeline()
        assert "nvarguscamerasrc" in pipeline
        assert "width=1920" in pipeline
        assert "height=1080" in pipeline
        assert "framerate=60/1" in pipeline
        assert "appsink" in pipeline
    
    @patch('cv2.VideoCapture')
    def test_open_success(self, mock_vc):
        """Test successful camera opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        result = camera._open()
        
        assert result is True
        assert camera._initialized is True
        mock_vc.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_open_failure(self, mock_vc):
        """Test camera opening failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        result = camera._open()
        
        assert result is False
        assert camera._initialized is False
    
    @patch('cv2.VideoCapture')
    def test_read_lazy_initialization(self, mock_vc):
        """Test lazy initialization on first read."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        success, frame = camera.read()
        
        assert success is True
        assert frame is not None
        assert camera._initialized is True
    
    @patch('cv2.VideoCapture')
    def test_read_after_init(self, mock_vc):
        """Test reading frame after initialization."""
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, test_frame)
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        camera._open()
        success, frame = camera.read()
        
        assert success is True
        assert frame is not None
        np.testing.assert_array_equal(frame, test_frame)
    
    def test_is_opened_false_when_not_initialized(self):
        """Test is_opened returns False when not initialized."""
        camera = CameraCapture()
        assert camera.is_opened() is False
    
    @patch('cv2.VideoCapture')
    def test_is_opened_true_when_opened(self, mock_vc):
        """Test is_opened returns True when camera is opened."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        camera._open()
        assert camera.is_opened() is True
    
    @patch('cv2.VideoCapture')
    def test_release(self, mock_vc):
        """Test camera release."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        camera = CameraCapture()
        camera._open()
        camera.release()
        
        mock_cap.release.assert_called_once()
        assert camera.cap is None
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_vc):
        """Test context manager usage."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_vc.return_value = mock_cap
        
        with CameraCapture() as camera:
            camera._open()
            assert camera.cap is not None
        
        mock_cap.release.assert_called_once()
