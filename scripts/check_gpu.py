#!/usr/bin/env python3
"""Check GPU capabilities in the container."""
import sys

print("=" * 60)
print("GPU Capability Check")
print("=" * 60)

# Check OpenCV
print("\n1. OpenCV:")
try:
    import cv2
    print(f"   Version: {cv2.__version__}")
    
    if hasattr(cv2, 'cuda'):
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   CUDA devices: {count}")
        if count > 0:
            print("   ✓ CUDA support available")
        else:
            print("   ✗ CUDA not available")
    else:
        print("   ✗ No cv2.cuda module")
        
    # Check DNN backends
    backends = []
    targets = []
    try:
        cv2.dnn.DNN_BACKEND_CUDA
        backends.append("CUDA")
    except:
        pass
    try:
        cv2.dnn.DNN_TARGET_CUDA_FP16
        targets.append("CUDA_FP16")
    except:
        pass
    
    if backends:
        print(f"   DNN Backends: {', '.join(backends)}")
    if targets:
        print(f"   DNN Targets: {', '.join(targets)}")
        
except Exception as e:
    print(f"   Error: {e}")

# Check TensorRT
print("\n2. TensorRT:")
try:
    import tensorrt as trt
    print(f"   Version: {trt.__version__}")
    print("   ✓ TensorRT available")
except ImportError:
    print("   ✗ Not installed")

# Check CUDA toolkit
print("\n3. CUDA Toolkit:")
import os
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home:
    print(f"   CUDA_HOME: {cuda_home}")
else:
    print("   No CUDA_HOME set")

# Check for nvcc
import subprocess
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()]
        if version_line:
            print(f"   {version_line[0].strip()}")
except:
    print("   nvcc not found")

print("\n" + "=" * 60)
