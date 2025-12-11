#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU health check and diagnostics for Jetson container."""
import sys
import os

def check_gpu_availability():
    """Check all GPU-related components."""
    results = {
        'opencv': False,
        'pytorch': False,
        'cuda': False
    }
    errors = []
    
    print("=" * 70)
    print("SYSTEM HEALTH CHECK - PyTorch GPU Face Detection")
    print("=" * 70)
    
    # Check OpenCV
    print("\n[1/3] Checking OpenCV...")
    try:
        import cv2
        print("  [OK] OpenCV {0} available".format(cv2.__version__))
        results['opencv'] = True
    except Exception as e:
        print("  [ERROR] OpenCV error: {0}".format(e))
        errors.append("OpenCV: {0}".format(e))
    
    # Check PyTorch
    print("\n[2/3] Checking PyTorch...")
    try:
        import torch
        print("  [OK] PyTorch {0} available".format(torch.__version__))
        results['pytorch'] = True
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("  [OK] CUDA available: {0}".format(torch.cuda.get_device_name(0)))
            print("  [OK] CUDA version: {0}".format(torch.version.cuda))
            props = torch.cuda.get_device_properties(0)
            print("  [OK] GPU memory: {0:.2f} GB".format(props.total_memory / 1024**3))
            print("  [OK] CUDA cores: {0}".format(props.multi_processor_count * 128))
            results['cuda'] = True
        else:
            print("  [WARN] CUDA not available (will use CPU fallback)")
            
    except Exception as e:
        print("  [ERROR] PyTorch error: {0}".format(e))
        errors.append("PyTorch: {0}".format(e))
    
    # Test GPU inference
    print("\n[3/3] Testing GPU inference...")
    if results['pytorch'] and results['cuda']:
        try:
            import torch
            import time
            
            device = torch.device('cuda:0')
            x = torch.randn(1, 3, 224, 224, device=device)
            
            start = time.time()
            y = torch.nn.functional.conv2d(x, torch.randn(64, 3, 3, 3, device=device))
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            
            print("  [OK] GPU inference test: {0:.2f}ms".format(elapsed))
            print("  [OK] PyTorch GPU acceleration READY")
        except Exception as e:
            print("  [ERROR] GPU inference failed: {0}".format(e))
            errors.append("GPU inference: {0}".format(e))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results['opencv'] and results['pytorch']:
        if results['cuda']:
            print("[OK] All components available")
            print("[OK] Face detection: PyTorch GPU")
            print("[OK] Performance: Real-time (30 FPS target)")
        else:
            print("[OK] System components available")
            print("[WARN] Face detection: CPU fallback (Haar Cascade)")
            print("[WARN] Performance: Reduced (~20 FPS)")
        return 0
    else:
        print("[ERROR] System check failed:")
        for error in errors:
            print("  - {0}".format(error))
        return 1

if __name__ == "__main__":
    exit_code = check_gpu_availability()
    print("=" * 70)
    sys.exit(exit_code)
