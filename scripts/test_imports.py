#!/usr/bin/env python3
"""Test TensorRT and pycuda imports."""
import sys

print("Testing imports...")
print("=" * 60)

# Test 1: TensorRT
print("\n1. TensorRT:")
try:
    import tensorrt as trt
    print(f"   ✓ SUCCESS - version {trt.__version__}")
except ImportError as e:
    print(f"   ✗ FAILED - {e}")
except Exception as e:
    print(f"   ✗ ERROR - {e}")

# Test 2: PyCUDA
print("\n2. PyCUDA:")
try:
    import pycuda
    print(f"   ✓ pycuda module found")
    import pycuda.driver as cuda
    print(f"   ✓ pycuda.driver imported")
    import pycuda.autoinit
    print(f"   ✓ pycuda.autoinit imported")
    print(f"   ✓ SUCCESS")
except ImportError as e:
    print(f"   ✗ FAILED - {e}")
except Exception as e:
    print(f"   ✗ ERROR - {e}")

# Test 3: Check Python path
print("\n3. Python Path:")
for p in sys.path[:5]:
    print(f"   {p}")

# Test 4: Check for TensorRT files
print("\n4. TensorRT files:")
import os
trt_paths = [
    '/usr/lib/python3.8/dist-packages/tensorrt',
    '/usr/local/lib/python3.8/dist-packages/tensorrt',
    '/opt/nvidia/trt/python'
]
for path in trt_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
    else:
        print(f"   ✗ Not found: {path}")

print("\n" + "=" * 60)
