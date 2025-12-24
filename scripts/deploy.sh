#!/usr/bin/env bash
# Deploy application files to Jetson Nano
# Usage: ./deploy.sh [user] [host] [remote_dir]
#
# Example: ./deploy.sh nvidia 192.168.1.67 jetson-webcam

set -e

USER_NAME=${1:-nvidia}
HOST=${2:-192.168.1.67}
REMOTE_DIR=${3:-jetson-webcam}

echo "=========================================="
echo "Deploying to ${USER_NAME}@${HOST}:${REMOTE_DIR}"
echo "=========================================="

echo "Syncing files..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'models_cache' \
  ./ \
  ${USER_NAME}@${HOST}:~/${REMOTE_DIR}/

echo ""
echo "✓ Deployment complete"
echo "Files synced to ${USER_NAME}@${HOST}:~/${REMOTE_DIR}"
echo ""
echo "Next steps:"
echo "  1. SSH to device: ssh ${USER_NAME}@${HOST}"
echo "  2. Build and start: cd ${REMOTE_DIR} && sudo docker-compose up -d --build"
echo "  3. Access stream: http://${HOST}:8000"
