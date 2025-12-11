#!/usr/bin/env python3
"""Unit tests for PyTorch GPU face detection."""
import unittest
import numpy as np
import cv2
import os
import sys

# Add app directory to path
sys.path.insert(0, '/app')

class TestGPUAvailability(unittest.TestCase):
    """Test GPU components availability."""
    
    def test_pytorch_import(self):
        """Test PyTorch can be imported."""
        try:
            import torch
            self.assertIsNotNone(torch.__version__)
            print(f"✓ PyTorch {torch.__version__}")
        except ImportError:
            self.fail("PyTorch not available")
    
    def test_cuda_availability(self):
        """Test CUDA availability in PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.assertIsNotNone(device_name)
                print(f"✓ CUDA available: {device_name}")
            else:
                print("⚠ CUDA not available (will use CPU fallback)")
        except Exception as e:
            self.fail(f"CUDA check failed: {e}")
    
    def test_opencv_import(self):
        """Test OpenCV can be imported."""
        self.assertIsNotNone(cv2.__version__)
        print(f"✓ OpenCV {cv2.__version__}")


class TestFaceDetectorTensorRT(unittest.TestCase):
    """Test PyTorch face detector functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize face detector once for all tests."""
        from face_detector_tensorrt import FaceDetectorTensorRT
        cls.detector = FaceDetectorTensorRT()
    
    def test_detector_initialization(self):
        """Test detector initializes without errors."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.backend)
        print(f"✓ Detector backend: {self.detector.backend}")
    
    def test_detector_backend(self):
        """Test detector backend type."""
        backend = self.detector.get_backend()
        # Should be PyTorch-GPU or Haar-CPU-Fallback
        self.assertIn(backend, ["PyTorch-GPU", "Haar-CPU-Fallback"])
        
        if backend == "PyTorch-GPU":
            print(f"✓ Using GPU acceleration: {backend}")
        else:
            print(f"⚠ Using CPU fallback: {backend}")
    
    def test_enable_disable(self):
        """Test enable/disable functionality."""
        self.detector.disable()
        self.assertFalse(self.detector.is_enabled())
        
        self.detector.enable()
        self.assertTrue(self.detector.is_enabled())
        print("✓ Enable/disable works")
    
    def test_detect_on_black_frame(self):
        """Test detection on a black frame (should return no faces)."""
        self.detector.enable()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        faces = self.detector.detect(black_frame)
        self.assertIsInstance(faces, list)
        # Black frame should have no faces
        self.assertEqual(len(faces), 0)
        print("✓ Black frame detection works")
    
    def test_detect_on_synthetic_frame(self):
        """Test detection on a synthetic frame with face-like pattern."""
        self.detector.enable()
        
        # Create a frame with random noise (more realistic than black)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = self.detector.detect(frame)
        self.assertIsInstance(faces, list)
        # Result should be a list (may or may not detect faces in noise)
        print(f"✓ Synthetic frame detection works (detected {len(faces)} faces)")
    
    def test_detect_returns_correct_format(self):
        """Test that detection returns correct format (x, y, w, h)."""
        self.detector.enable()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = self.detector.detect(frame)
        
        for face in faces:
            self.assertIsInstance(face, tuple)
            self.assertEqual(len(face), 4)  # (x, y, w, h)
            x, y, w, h = face
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
        
        print(f"✓ Detection format correct for {len(faces)} faces")
    
    def test_detect_when_disabled(self):
        """Test that detection returns empty when disabled."""
        self.detector.disable()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = self.detector.detect(frame)
        self.assertEqual(len(faces), 0)
        print("✓ Disabled detector returns empty list")
    
    def test_frame_skipping(self):
        """Test frame skipping optimization."""
        self.detector.enable()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # First detection
        faces1 = self.detector.detect(frame)
        frame_count1 = self.detector.frame_count
        
        # Second detection (might be skipped)
        faces2 = self.detector.detect(frame)
        frame_count2 = self.detector.frame_count
        
        # Frame count should increment
        self.assertEqual(frame_count2, frame_count1 + 1)
        print("✓ Frame skipping works")


class TestTensorRTInference(unittest.TestCase):
    """Test TensorRT inference engine directly."""
    
    def test_onnx_model_download(self):
        """Test ONNX model can be downloaded or is present."""
        from face_detector_tensorrt import FaceDetectorTensorRT
        
        detector = FaceDetectorTensorRT()
        model_path = detector._get_onnx_model()
        
        if model_path:
            self.assertTrue(os.path.exists(model_path))
            print(f"✓ ONNX model available at: {model_path}")
        else:
            print("⚠ ONNX model download skipped (expected in some configs)")
    
    def test_tensorrt_engine_creation(self):
        """Test TensorRT engine can be created (if GPU available)."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # If we get here, TensorRT and CUDA are available
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            
            self.assertIsNotNone(builder)
            print("✓ TensorRT Builder can be created")
            
        except ImportError:
            self.skipTest("TensorRT or PyCUDA not available")


def run_tests():
    """Run all tests and return exit code."""
    print("=" * 70)
    print("FACE DETECTION GPU UNIT TESTS")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGPUAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceDetectorTensorRT))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorRTInference))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    # Return 0 if all tests passed
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
