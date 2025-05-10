# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3-setuptools \
    python3-wheel \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    cmake \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    libegl1 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libglib2.0-0 \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git .

# Install PyTorch with CUDA support (following the exact version in INSTALL.md)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install project dependencies as specified in INSTALL.md
RUN pip3 install --no-cache-dir \
    diffusers==0.25.0 \
    accelerate \
    einops \
    hydra-core \
    wandb \
    pymeshlab \
    open3d \
    transforms3d

# Install MuJoCo
RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xvzf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

# Set MuJoCo environment variables
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_GL=egl

# Install mujoco-py with the specific version
RUN pip3 install --no-cache-dir 'mujoco-py<2.2,>=2.1'

# Fix potential mujoco-py installation issues
RUN mkdir -p /usr/lib/nvidia-000
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia-000

# Clone and install pointnet2 exactly as in INSTALL.md
RUN mkdir -p third_party \
    && cd third_party \
    && git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git \
    && cd Pointnet2_PyTorch \
    && pip install -e .

# Clone and install clip exactly as in INSTALL.md
RUN pip3 install --no-cache-dir ftfy regex tqdm \
    && pip3 install --no-cache-dir git+https://github.com/openai/CLIP.git

# Install the project itself
WORKDIR /app
RUN pip3 install --no-cache-dir -e .

# Set environment variables for GPU use
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PYOPENGL_PLATFORM egl

# Default command
CMD ["/bin/bash"]