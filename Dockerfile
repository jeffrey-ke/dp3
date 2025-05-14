# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

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
# Miniconda installation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    source /opt/conda/etc/profile.d/conda.sh && \ 
    conda create -n dp3 python=3.8 -y
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "dp3", "/bin/bash", "-c"]

# Set working directory
WORKDIR /root/ws

# Clone the repository
RUN ls -R /root/ws | head -20     # shows the repo layout during build
RUN git clone https://github.com/jeffrey-ke/dp3.git

# Install PyTorch with CUDA support (following the exact version in INSTALL.md)
WORKDIR /root/ws/dp3/3D-Diffusion-Policy
RUN pip install --no-cache-dir torch==2.0.1+cu118 \
                                torchvision==0.15.2+cu118 \
                                --extra-index-url https://download.pytorch.org/whl/cu118
# install dp3
RUN pip install -e 3D-Diffusion-Policy
RUN pip install -e vggt
# install mujoco
WORKDIR /root/.mujoco
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
RUN tar -xvzf mujoco210.tar.gz
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 \
    MUJOCO_GL=egl

WORKDIR /root/ws/dp3/3D-Diffusion-Policy/third_party
RUN ["pip", "install", "-e", "mujoco-py-2.1.2.14"]
# install sim env
RUN pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0
RUN pip install -e gym-0.21.0 
RUN pip install -e dexart-release
RUN pip install -e Metaworld rrl-dependencies/mj_envs rrl-dependencies/mjrl 
COPY vrl3_ckpts.zip VRL3/
RUN unzip VRL3/vrl3_ckpts.zip -d VRL3 && mv VRL3/vrl3_ckpts VRL3/ckpts
COPY dexart_assets.zip dexart-release/
RUN unzip dexart-release/dexart_assets.zip -d dexart-release

# install pytorch3d
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install some necessary packages
RUN pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor huggingface_hub==0.25.2

# Install project dependencies as specified in INSTALL.md
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    accelerate \
    einops \
    hydra-core \
    wandb \
    pymeshlab \
    open3d \
    transforms3d

# Default command
RUN echo "conda activate dp3" >> ~/.bashrc
CMD ["/bin/bash"]
