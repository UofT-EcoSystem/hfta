FROM nvcr.io/nvidia/pytorch:20.08-py3

RUN apt update && apt install -y --no-install-recommends \
		python2.7 python-dev python-pip python-setuptools \
		python3 python3-dev python3-setuptools python3-pip \ 
		vim git curl tmux wget unzip \
		ca-certificates apt-utils \
		libtinfo-dev zlib1g-dev libjpeg-dev libpng-dev libglib2.0-0 && \
		python2 -m pip install wheel psutil pandas && \
		/opt/conda/bin/pip3 install lmdb pandas matplotlib && \
		rm -rf /var/lib/apt/lists/*


COPY third_party/dcgm/datacenter-gpu-manager_2.0.10_amd64.deb dcgm.deb
RUN apt-get install -y --no-install-recommends ./dcgm.deb
