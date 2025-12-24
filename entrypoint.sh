#!/bin/bash
# Container entrypoint with GPU validation

set -e

echo "======================================================================="
echo "Jetson Vision Container Starting"
echo "======================================================================="

# Run GPU health check
echo ""
echo "Running GPU health check..."
python3 /app/gpu_health_check.py
GPU_CHECK_EXIT=$?

if [ $GPU_CHECK_EXIT -eq 0 ]; then
    echo ""
    echo "✓ GPU health check PASSED"
    echo "  GPU-accelerated face detection is READY"
else
    echo ""
    echo "⚠ GPU health check FAILED"
    echo "  Face detection will use CPU fallback"
fi

# Optional: Run unit tests (can be disabled with env var)
if [ "${RUN_TESTS_ON_STARTUP:-false}" = "true" ]; then
    echo ""
    echo "Running unit tests..."
    python3 /app/test_face_detection.py
    TEST_EXIT=$?
    
    if [ $TEST_EXIT -eq 0 ]; then
        echo ""
        echo "✓ Unit tests PASSED"
    else
        echo ""
        echo "⚠ Unit tests FAILED"
        if [ "${FAIL_ON_TEST_ERROR:-false}" = "true" ]; then
            echo "ERROR: Tests failed and FAIL_ON_TEST_ERROR=true"
            exit 1
        fi
    fi
fi

echo ""
echo "======================================================================="
echo "Starting application..."
echo "======================================================================="
echo ""

# Execute the main command (uvicorn)
exec "$@"
