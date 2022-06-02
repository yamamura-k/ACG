FROM nvidia/cuda:11.0-devel-ubuntu20.04

WORKDIR /AutoCG
COPY . .

ENV DEBIAN_FRONTEND=noninteractive
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV GIT_DISCOVERY_ACROSS_FILESYSTEM=1

RUN apt-get update
RUN apt-get install -y python3 python3-pip git libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/fra31/auto-attack

RUN cd src/utils/cluster_coef_c \
    && python3 setup.py build_ext -i

RUN echo "alias python='python3'" >> ~/.bashrc

CMD ["/bin/bash"]
