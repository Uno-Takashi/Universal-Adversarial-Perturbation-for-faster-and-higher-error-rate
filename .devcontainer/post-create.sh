#!/usr/bin/env bash
# Provision the project environment once the container is created.
set -euo pipefail

echo "==> uv $(uv --version)"

# The base image already provides CUDA and cuDNN system-wide, so the plain `tensorflow`
# wheel finds them and uses the GPU. The `cuda` extra (tensorflow[and-cuda]) is therefore
# not needed here -- it would add ~4 GB of duplicate CUDA wheels, and any later bare
# `uv run` would strip them again, since uv resyncs to the default extras.
uv sync --group dev

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
