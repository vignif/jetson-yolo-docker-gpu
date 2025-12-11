#!/bin/bash
# Complete rebuild and GPU testing script

set -e

echo "======================================================================="
echo "Jetson Vision - Complete Rebuild and GPU Testing"
echo "======================================================================="
echo ""

# Connect to Jetson
JETSON_HOST="nvidia@192.168.1.67"
JETSON_DIR="jetson-webcam"

echo "Step 1: Rebuilding container..."
ssh -t $JETSON_HOST "cd $JETSON_DIR && sudo docker-compose down && sudo docker-compose up -d --build"

echo ""
echo "Step 2: Waiting for container to start (15 seconds)..."
sleep 15

echo ""
echo "Step 3: Running GPU health check..."
ssh -t $JETSON_HOST "cd $JETSON_DIR && sudo docker-compose exec -T jetson-vision python3 /app/gpu_health_check.py"

echo ""
echo "Step 4: Running unit tests..."
ssh -t $JETSON_HOST "cd $JETSON_DIR && sudo docker-compose exec -T jetson-vision python3 /app/test_face_detection.py"

echo ""
echo "Step 5: Checking GPU status via API..."
ssh $JETSON_HOST "curl -s http://localhost:8000/api/gpu-status | python3 -m json.tool"

echo ""
echo "======================================================================="
echo "Testing Complete!"
echo "======================================================================="
echo ""
echo "Access the application at: http://192.168.1.67:8000"
echo "GPU Status API: http://192.168.1.67:8000/api/gpu-status"
echo "Run Tests API: http://192.168.1.67:8000/api/run-gpu-tests"
echo ""
