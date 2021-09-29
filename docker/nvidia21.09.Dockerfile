FROM nvcr.io/nvidia/pytorch:21.09-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
		  vim git curl tmux wget unzip apt-utils software-properties-common \
		  libglib2.0-0 && \
		rm -rf /var/lib/apt/lists/*

RUN conda install -y lmdb python-lmdb pandas matplotlib psutil tqdm && \
    conda clean -ya

RUN pip install --no-cache-dir plyfile hyperopt

# Install DCGM.
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && apt-get install -y datacenter-gpu-manager
