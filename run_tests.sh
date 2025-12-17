#!/bin/bash
# Run tests inside the Docker container
# Execute this from the Jetson device: ./run_tests.sh

docker-compose exec jetson-vision python3 -m pytest tests/ -v --tb=short
