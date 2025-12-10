#!/usr/bin/env zsh

# Deploy and run Jetson camera streaming app on remote device
# Usage: ./deploy.sh [user] [host] [remote_dir]
# Defaults: user=jetson, host=192.168.1.67, remote_dir=jetson-webcam

set -e

USER_NAME=${1:-nvidia}
HOST=${2:-192.168.1.67}
REMOTE_DIR=${3:-jetson-webcam}

echo "Deploying to ${USER_NAME}@${HOST}:${REMOTE_DIR}"

# 1) Create remote dir and sync files (no sudo required)
ssh -o StrictHostKeyChecking=no ${USER_NAME}@${HOST} "mkdir -p ~/${REMOTE_DIR}"

echo "Syncing files to remote (no delete to avoid permission issues)..."
rsync -avz \
  Dockerfile docker-compose.yml requirements.txt \
  app \
  ${USER_NAME}@${HOST}:~/${REMOTE_DIR}/

# 2) Run all privileged operations in ONE sudo session
echo "Running setup with one sudo prompt using askpass..."
read -s "?Enter remote sudo password: " SUDO_PW
echo

# Use ssh -tt to force TTY and pipe password to sudo -S
ssh -tt -o StrictHostKeyChecking=no ${USER_NAME}@${HOST} "bash -s" <<EOS
set -e
REMOTE_DIR="\$HOME/${REMOTE_DIR}"

# Cache sudo timestamp
echo "${SUDO_PW}" | sudo -S -p '' true

# Ensure Docker
# echo "${SUDO_PW}" | sudo -S -p '' apt-get update
# echo "${SUDO_PW}" | sudo -S -p '' apt-get install -y docker.io
# echo "${SUDO_PW}" | sudo -S -p '' systemctl enable docker
# echo "${SUDO_PW}" | sudo -S -p '' systemctl start docker

# Ensure docker-compose
# if ! command -v docker-compose >/dev/null 2>&1; then
#  echo "${SUDO_PW}" | sudo -S -p '' apt-get install -y docker-compose || {
#    echo "${SUDO_PW}" | sudo -S -p '' apt-get install -y python3-pip
#    echo "${SUDO_PW}" | sudo -S -p '' pip3 install docker-compose
#  }
# fi

# Add current user to docker group
# echo "${SUDO_PW}" | sudo -S -p '' usermod -aG docker "\$USER" || true

# Install NVIDIA Container Runtime if missing
# distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
# if ! dpkg -s nvidia-docker2 >/dev/null 2>&1; then
#   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
#   curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
#   echo "${SUDO_PW}" | sudo -S -p '' apt-get update
#   echo "${SUDO_PW}" | sudo -S -p '' apt-get install -y nvidia-docker2
#   echo "${SUDO_PW}" | sudo -S -p '' systemctl restart docker
# fi

# Rebuild and start containers detached
# cd "\$REMOTE_DIR"
# echo "${SUDO_PW}" | sudo -S -p '' chown -R "\$USER":"\$USER" "\$REMOTE_DIR" || true
# echo "${SUDO_PW}" | sudo -S -p '' rm -rf "\$REMOTE_DIR/app/__pycache__" || true
# echo "${SUDO_PW}" | sudo -S -p '' docker-compose down || true
# echo "${SUDO_PW}" | sudo -S -p '' docker-compose up -d --build
# echo "Stack started detached. To view logs: docker-compose logs -f jetson-vision"
EOS

echo "Deployment complete. Access the app at: http://${HOST}:8000"
