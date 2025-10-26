# Use CUDA 12.1.1 development image compatible with driver 530.30.02
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Non-interactive apt & pip noise off
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PROGRESS_BAR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Avoid docker-clean surprises (ignore if absent)
RUN rm -f /etc/apt/apt.conf.d/docker-clean || true

# System packages (minimal, no interactive recommends)
RUN set -eux; \
  apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    zlib1g-dev \
    libgl1 \
    vim \
    git \
    gnupg \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# --- Bake the key: COPY pre-downloaded keyring, add repo, install Python 3.11 ---
COPY deadsnakes.gpg /etc/apt/keyrings/deadsnakes.gpg
RUN set -eux; \
  chmod a+r /etc/apt/keyrings/deadsnakes.gpg; \
  echo 'deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main' \
    > /etc/apt/sources.list.d/deadsnakes-ppa.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
  && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded get-pip.py (baked on host) into the image
COPY get-pip.py /tmp/get-pip.py

# Set Python 3.11 as default and install pip + uv (no venv)
RUN set -eux; \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1; \
  update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1; \
  python3.11 /tmp/get-pip.py; \
  ln -sf /usr/local/bin/pip /usr/bin/pip; \
  python3.11 -m pip install --no-cache-dir uv; \
  rm -f /tmp/get-pip.py

# ---- Python packages ----
# Install PyTorch first (with CUDA 12.1 support)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ML/AI packages in logical groups to optimize layer caching
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    datasets==3.1.0 \
    accelerate==1.1.1 \
    bitsandbytes==0.44.1 \
    trl==0.12.1 \
    peft==0.13.2 \
    sentencepiece \
    intel_extension_for_pytorch

# Install vLLM separately (may have specific dependency requirements)
RUN pip install --no-cache-dir vllm==0.5.0

# Install data science and web packages
RUN pip install --no-cache-dir \
    numpy pandas matplotlib scikit-learn statsmodels \
    requests beautifulsoup4 fastapi \
    langchain-openai mlflow

# Install Jupyter ecosystem last
RUN pip install --no-cache-dir jupyter jupyter_server jupyterlab

# Workdir
WORKDIR /home

# CUDA paths
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Jupyter Server config (no auth, /home as root)
RUN set -eux; \
  jupyter server --generate-config; \
  printf "c.ServerApp.token = ''\n"                >> /root/.jupyter/jupyter_server_config.py; \
  printf "c.ServerApp.password = ''\n"             >> /root/.jupyter/jupyter_server_config.py; \
  printf "c.ServerApp.password_required = False\n" >> /root/.jupyter/jupyter_server_config.py; \
  printf "c.ServerApp.root_dir = '/home'\n"        >> /root/.jupyter/jupyter_server_config.py; \
  printf "c.ServerApp.preferred_dir = '/home'\n"   >> /root/.jupyter/jupyter_server_config.py

# Expose Jupyter
EXPOSE 8888

# Run Jupyter Server (UI at /lab)
CMD jupyter server --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.default_url=/lab --ServerApp.root_dir=/home