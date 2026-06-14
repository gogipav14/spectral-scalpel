# CodeOcean / RCR reproducibility image for spectral-scalpel.
#
# Base: NVIDIA CUDA 13 Ubuntu 22.04 runtime + cuDNN. The CUDA-13
# stream is what the reference workstation (Intel Core Ultra 5 225F,
# RTX 5060, WSL2) uses, and matches the JAX cuda13 wheel and
# cupy-cuda13x extras pinned in pyproject.toml.
#
# CodeOcean: place this Dockerfile at the capsule's environment/
# directory or point the capsule build at this file via the
# "Reproducible Capsule" Docker workflow. The reproduce.sh below is
# the capsule's intended `run` script.

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# JAX VRAM hygiene defaults (overridden by reproduce.sh if desired).
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

# System dependencies: Python 3.12, build tools for any source-only
# wheels, and git so the capsule can pull updates if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

# Repository checkout. CodeOcean typically mounts the user's code
# under /code; we honor that convention so the same image works for
# `docker run -v $PWD:/code` style local invocations.
WORKDIR /code
COPY .. /code

# Install the library with the full multi-backend extras (JAX CUDA,
# PyTorch, CuPy, mpmath, viz, dev). The Julia extra is intentionally
# omitted from the default image because the Julia reference path is
# documented as a portability reference, not part of the dispatched
# matrix.
RUN python3 -m pip install --no-cache-dir -e ".[jax,torch,cupy,viz,mpmath,dev]"

# Default command: full RCR reproduction (multi-backend, GPU-enabled).
# CodeOcean's "Reproducible Run" button invokes this.
CMD ["bash", "reproducibility/reproduce.sh"]
