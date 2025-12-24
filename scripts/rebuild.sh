#!/usr/bin/env bash
# Rebuild and restart container on Jetson Nano
# Usage: ./rebuild.sh [user@host]
#
# Example: ./rebuild.sh nvidia@192.168.1.67

set -e

JETSON_HOST=${1:-nvidia@192.168.1.67}

echo "=========================================="
echo "Rebuilding container on ${JETSON_HOST}"
echo "=========================================="

ssh ${JETSON_HOST} << 'ENDSSH'
cd jetson-webcam
echo "→ Stopping container..."
sudo docker-compose down
echo "→ Rebuilding image..."
sudo docker-compose build
echo "→ Starting container..."
sudo docker-compose up -d
echo "→ Waiting for startup..."
sleep 12
echo ""
echo "=========================================="
echo "Container logs (last 50 lines):"
echo "=========================================="
sudo docker-compose logs --tail=50
ENDSSH

echo ""
echo "✓ Rebuild complete"
echo "Access: http://${JETSON_HOST#*@}:8000"
