#!/bin/bash
echo "Rebuilding container on Jetson..."
ssh nvidia@192.168.1.67 << 'ENDSSH'
cd jetson-webcam
echo "Stopping container..."
docker-compose down
echo "Rebuilding image with MPI support..."
docker-compose build
echo "Starting container..."
docker-compose up -d
echo "Waiting for startup..."
sleep 12
echo "Container logs:"
docker-compose logs --tail=120
ENDSSH
