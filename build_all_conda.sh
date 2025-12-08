#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install dependencies
# Using PyTorch with CUDA 12.4 for compatibility with CUDA 12.8
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
# git clone https://github.com/NVlabs/FoundationPose.git

# pip install gdown

git clone https://github.com/wkentaro/gdown
sed -i 's/MAX_NUMBER_FILES = 50/MAX_NUMBER_FILES = 1000/' gdown/gdown/download_folder.py
cd gdown && pip install -e . --no-cache-dir

cd ..

pip install ruamel.yaml

# git clone https://github.com/RoboticManipulation/FoundationPose.git

# cd FoundationPose
# Create the weights directory and download the pretrained weights from FoundationPose
# gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37 
# gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45


## weights
if [ ! -d "weights" ] && [ ! -d "1jocuP_wFByHw6nME0ZdLDV8HVsRksZNL" ]; then
    gdown --folder  https://drive.google.com/drive/folders/1jocuP_wFByHw6nME0ZdLDV8HVsRksZNL?usp=sharing
else
    echo "Weights folder already exists, skipping download."
fi
## demo_data
if [ ! -d "demo_data" ] && [ ! -d "1PYIuQ6Q6IsF3rpqu5Hclln6Iok6qbUI0" ]; then
    gdown --folder https://drive.google.com/drive/folders/1PYIuQ6Q6IsF3rpqu5Hclln6Iok6qbUI0?usp=sharing
else
    echo "Demo data folder already exists, skipping download."
fi

# Install pybind11
cd ${PROJ_ROOT} && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    sudo make -j6 && sudo make install

# Install Eigen
cd ${PROJ_ROOT} && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    sudo make install

# Clone and install nvdiffrast
# nvdiffrast needs PyTorch visible during build, so we must disable build isolation.
cd ${PROJ_ROOT} && ( [ -d nvdiffrast ] || git clone https://github.com/NVlabs/nvdiffrast )
cd ${PROJ_ROOT}/nvdiffrast && pip install . --no-build-isolation

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(nproc-1)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so # && \
# python3 -m pip install -e . --no-build-


## ERROR RESOLUTION for mycuda installation
source ~/miniconda3/bin/activate foundationpose_ros && pip uninstall torch torchvision torchaudio -y && pip cache purge
source ~/miniconda3/bin/activate foundationpose_ros && pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
source ~/miniconda3/bin/activate foundationpose_ros && python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
source ~/miniconda3/bin/activate foundationpose_ros && cd ${PROJ_ROOT}/bundlesdf/mycuda && rm -rf build *egg* *.so && export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}} && python3 -m pip install -e . --no-build-isolation

cd ${PROJ_ROOT}
