#!/usr/bin/env bash
# Run tests inside the Docker container
# Execute this directly on the Jetson device
# Usage: ./run_tests.sh

set -e

echo "=========================================="
echo "Running tests in container..."
echo "=========================================="

sudo docker-compose exec jetson-vision python3 -m pytest tests/ -v --tb=short

echo ""
echo "✓ Tests complete"
