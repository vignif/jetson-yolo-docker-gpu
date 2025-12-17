#!/bin/bash
# Deploy and run tests, then restart if tests pass

set -e

echo "Deploying to Jetson..."
./deploy.sh

echo ""
echo "Running tests on Jetson..."
./test.sh

TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "Tests passed! Restarting application..."
    ssh nvidia@192.168.1.67 "cd jetson-webcam && docker-compose restart"
    echo "✓ Deployment complete!"
else
    echo ""
    echo "✗ Tests failed! Application not restarted."
    exit 1
fi
