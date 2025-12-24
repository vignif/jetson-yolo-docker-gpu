# Scripts

Utility scripts for deployment, building, and testing the Jetson camera streaming application.

## Deployment & Build

### `deploy.sh`
Deploy application files to Jetson Nano device.

**Usage:**
```bash
./scripts/deploy.sh [user] [host] [remote_dir]
```

**Example:**
```bash
./scripts/deploy.sh nvidia 192.168.1.67 jetson-webcam
```

### `rebuild.sh`
Rebuild and restart the Docker container on Jetson.

**Usage:**
```bash
./scripts/rebuild.sh [user@host]
```

**Example:**
```bash
./scripts/rebuild.sh nvidia@192.168.1.67
```

## Testing

### `run_tests.sh`
Run pytest test suite inside the container. Execute this **on the Jetson device**.

**Usage:**
```bash
./scripts/run_tests.sh
```

### `check_gpu.py`
Check GPU, CUDA, and TensorRT availability on the system.

**Usage:**
```bash
python3 scripts/check_gpu.py
```

### `test_imports.py`
Test TensorRT and PyCUDA imports with detailed diagnostics.

**Usage:**
```bash
python3 scripts/test_imports.py
```

### `run_tests_local.py`
Run basic local tests without requiring a Jetson device.

**Usage:**
```bash
python3 scripts/run_tests_local.py
```

## Making Scripts Executable

```bash
chmod +x scripts/*.sh
```
