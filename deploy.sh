#!/usr/bin/env zsh

# Deploy files to Jetson device
# Usage: ./deploy.sh [user] [host] [remote_dir]

set -e

USER_NAME=${1:-nvidia}
HOST=${2:-192.168.1.67}
REMOTE_DIR=${3:-jetson-webcam}

echo "Deploying to ${USER_NAME}@${HOST}:${REMOTE_DIR}"

# Create remote directory and sync all files
# ssh -o StrictHostKeyChecking=no ${USER_NAME}@${HOST} "mkdir -p ~/${REMOTE_DIR}"

echo "Syncing files..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  ./ \
  ${USER_NAME}@${HOST}:~/${REMOTE_DIR}/

echo "✓ Deployment complete"
echo "Files synced to ${USER_NAME}@${HOST}:~/${REMOTE_DIR}"
echo ""
echo "To rebuild and restart on the device, run:"
echo "  ssh ${USER_NAME}@${HOST} 'cd ${REMOTE_DIR} && sudo docker-compose down && sudo docker-compose up -d --build'"
