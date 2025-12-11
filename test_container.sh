#!/bin/bash
# Quick test script to run inside container
echo "Testing inside container..."
sudo docker-compose exec jetson-vision python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import tensorrt as trt
    print(f'TensorRT: {trt.__version__} ✓')
except Exception as e:
    print(f'TensorRT: FAILED - {e}')
try:
    import pycuda
    print(f'PyCUDA: ✓')
except Exception as e:
    print(f'PyCUDA: FAILED - {e}')
"
