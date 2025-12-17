#!/bin/bash
# Run tests on Jetson before deployment

echo "=========================================="
echo "Running tests on Jetson Nano..."
echo "=========================================="

ssh nvidia@192.168.1.67 "cd jetson-webcam && docker-compose exec -T jetson-vision python3 -m pytest tests/ -v --tb=short"

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ Tests failed!"
    echo "=========================================="
    exit $TEST_EXIT_CODE
fi
