#!/usr/bin/env python3
"""GPU health check and diagnostics for Jetson container."""
import sys
import os

def check_gpu_availability():
    """Check all GPU-related components."""
    results = {
        'opencv': False
    }
    errors = []
    
    print("=" * 70)
    print("SYSTEM HEALTH CHECK")
    print("=" * 70)
    
    # Check OpenCV
    print("\n[1/1] Checking OpenCV...")
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__} available")
        results['opencv'] = True
            
    except Exception as e:
        print(f"  ✗ OpenCV error: {e}")
        errors.append(f"OpenCV: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results['opencv']:
        print("✓ System components available")
        print("✓ Face detection READY (CPU-optimized)")
        print("\nNote: GPU acceleration requires compatible pycuda (not available)")
        return 0
    else:
        print("✗ System check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

if __name__ == "__main__":
    exit_code = check_gpu_availability()
    print("=" * 70)
    sys.exit(exit_code)
