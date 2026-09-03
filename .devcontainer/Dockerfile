# CUDA 12.6 + cuDNN 9 devel image: gives nvcc and the CUDA headers inside the container.
# The Python-side CUDA libraries come from the `tensorflow[and-cuda]` wheels, so this base
# image mainly provides the toolchain and lets `nvidia-smi` work through the container runtime.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Ubuntu 24.04 images ship a uid-1000 "ubuntu" user; reuse that slot for the dev user.
RUN if id -u ${USER_UID} >/dev/null 2>&1; then \
        existing="$(id -un ${USER_UID})"; \
        if [ "${existing}" != "${USERNAME}" ]; then \
            usermod -l ${USERNAME} "${existing}" \
            && groupmod -n ${USERNAME} "$(id -gn ${USER_UID})" \
            && usermod -d /home/${USERNAME} -m ${USERNAME}; \
        fi; \
    else \
        groupadd --gid ${USER_GID} ${USERNAME} \
        && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; \
    fi

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        libgl1 \
        libglib2.0-0 \
        locales \
        openssh-client \
        sudo \
        unzip \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    # Keep TensorFlow's startup logging to warnings and above.
    TF_CPP_MIN_LOG_LEVEL=1 \
    # Install uv-managed interpreters and the venv in predictable places.
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/home/${USERNAME}/.local/share/uv/python \
    PATH=/workspaces/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/.venv/bin:/home/${USERNAME}/.local/bin:${PATH}

USER ${USERNAME}

# Install uv (pinned) for the dev user.
ARG UV_VERSION=0.12.5
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
