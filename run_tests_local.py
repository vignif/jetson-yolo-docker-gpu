"""Test runner - execute on Jetson via SSH."""
import subprocess
import sys

def run_tests_on_jetson():
    """Run tests on the Jetson device via docker-compose."""
    print("Running tests on Jetson Nano...")
    print("=" * 60)
    
    result = subprocess.run([
        "ssh", "nvidia@192.168.1.67",
        "cd jetson-webcam && docker-compose exec -T jetson-vision python3 -m pytest tests/ -v --tb=short"
    ])
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests_on_jetson()
    sys.exit(exit_code)
