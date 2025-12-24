# Contributing to Jetson Camera Streaming

Thank you for your interest in contributing! This project aims to provide a reliable, high-performance camera streaming solution for NVIDIA Jetson devices.

## Development Setup

### Prerequisites

- NVIDIA Jetson Nano with JetPack 4.x
- Raspberry Pi Camera Module v2
- Docker and Docker Compose
- Basic knowledge of Python, FastAPI, and GStreamer

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/jetson-webcam.git
   cd jetson-webcam
   ```

3. Deploy to your Jetson device:
   ```bash
   ./scripts/deploy.sh nvidia YOUR_JETSON_IP jetson-webcam
   ```

4. Build and run:
   ```bash
   ssh nvidia@YOUR_JETSON_IP
   cd jetson-webcam
   sudo docker-compose up -d --build
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

Run the test suite before submitting:

```bash
# On Jetson device
./scripts/run_tests.sh

# Or from development machine
ssh nvidia@YOUR_JETSON_IP "cd jetson-webcam && sudo docker-compose exec -T jetson-vision python3 -m pytest tests/ -v"
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and test thoroughly
3. Commit with clear messages: `git commit -m "Add feature: description"`
4. Push to your fork: `git push origin feature/my-feature`
5. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Testing performed
   - Screenshots (if UI changes)

## Areas for Contribution

- **Performance**: Optimize frame processing and encoding
- **Features**: Add new detection models or streaming protocols
- **Documentation**: Improve guides and add examples
- **Testing**: Expand test coverage
- **Hardware Support**: Test on other Jetson models
- **Bug Fixes**: Address issues in the issue tracker

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Discussion of potential changes

Thank you for contributing!
