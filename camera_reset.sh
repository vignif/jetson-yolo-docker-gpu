#!/bin/bash
# Camera reset script - run on Jetson when camera is locked
# Usage: ./camera_reset.sh

echo "=========================================="
echo "Resetting Camera System"
echo "=========================================="

echo ""
echo "[1/4] Stopping Docker container..."
docker-compose stop

echo ""
echo "[2/4] Checking for processes using camera..."
sudo fuser -v /dev/video0 2>/dev/null || echo "No processes found on /dev/video0"

echo ""
echo "[3/4] Restarting nvargus daemon..."
sudo systemctl restart nvargus-daemon
sleep 2

echo ""
echo "[4/4] Checking camera device..."
if [ -e /dev/video0 ]; then
    echo "✓ /dev/video0 exists"
    ls -l /dev/video0
else
    echo "✗ /dev/video0 not found"
fi

echo ""
echo "=========================================="
echo "Camera reset complete!"
echo "Now run: docker-compose up -d"
echo "=========================================="
