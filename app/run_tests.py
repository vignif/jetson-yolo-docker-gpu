"""Test runner script - runs all tests before starting the application."""
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)


def run_tests():
    """Run all tests and return success status."""
    print("=" * 60)
    print("RUNNING TESTS BEFORE APPLICATION STARTUP")
    print("=" * 60)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=False,
            text=True
        )
        
        print("=" * 60)
        if result.returncode == 0:
            print("✓ ALL TESTS PASSED")
            print("=" * 60)
            return True
        else:
            print("✗ TESTS FAILED")
            print("=" * 60)
            return False
            
    except Exception as e:
        print("✗ ERROR RUNNING TESTS:", str(e))
        print("=" * 60)
        return False


if __name__ == "__main__":
    # Direct execution - run tests
    print("\nRunning tests in container...")
    success = run_tests()
    sys.exit(0 if success else 1)
