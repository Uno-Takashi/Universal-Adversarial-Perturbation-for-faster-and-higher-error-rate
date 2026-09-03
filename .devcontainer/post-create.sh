#!/usr/bin/env bash
# Provision the project environment once the container is created.
set -euo pipefail

echo "==> uv $(uv --version)"

# Install the project with the CUDA-enabled TensorFlow build plus the dev tooling.
# On a machine without an NVIDIA GPU this still works: the CUDA wheels install but stay unused.
uv sync --extra cuda --group dev

echo
echo "==> GPU visibility"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
    echo "==> TensorFlow device check"
    uv run python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow {tf.__version__}')
print(f'GPUs visible to TensorFlow: {gpus or \"none\"}')
" || echo "TensorFlow could not be queried; see the output above."
else
    cat <<'MSG'
nvidia-smi is unavailable, so this container has no GPU access.
TensorFlow will run on the CPU. To enable the GPU you need, on the host:
  - a recent NVIDIA driver
  - the NVIDIA Container Toolkit (nvidia-container-toolkit)
  - Docker configured with the nvidia runtime
On WSL2, install the NVIDIA driver on Windows (not inside WSL) and enable
GPU support in Docker Desktop.
MSG
fi

echo
echo "==> Ready. Try: uv run pytest -q"
