FROM nvcr.io/nvidia/pytorch:20.06-py3

RUN apt update && \
    apt install -y --no-install-recommends \
		  python3 python3-dev python3-setuptools python3-pip \ 
		  vim git curl tmux wget unzip \
		  ca-certificates apt-utils \
		  libtinfo-dev zlib1g-dev libjpeg-dev libpng-dev libglib2.0-0 && \
		rm -rf /var/lib/apt/lists/*

RUN conda install -y lmdb python-lmdb pandas matplotlib psutil tqdm && \
    conda clean -ya

RUN pip install --no-cache-dir plyfile hyperopt

COPY third_party/dcgm/datacenter-gpu-manager_2.0.10_amd64.deb dcgm.deb
RUN apt-get install -y --no-install-recommends ./dcgm.deb
