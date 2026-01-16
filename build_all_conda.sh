#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set CUDA environment variables for CUDA 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Clean any existing PyTorch installation to avoid conflicts
pip uninstall torch torchvision torchaudio -y
pip cache purge

# Install dependencies
# Using PyTorch with CUDA 12.4 for compatibility with CUDA 12.8
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA version
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check if PyTorch has the correct CUDA version
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
if [ "$TORCH_CUDA" != "12.4" ]; then
    echo "ERROR: PyTorch is not using CUDA 12.4! Found CUDA $TORCH_CUDA"
    echo "Please ensure PyTorch is installed with CUDA 12.4 support before continuing."
    exit 1
fi
echo "✓ PyTorch CUDA version verified"
### -- The following command takes a lot of time , so please be patient.
echo " The following command takes a lot of time , so please be patient.pyth..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
python -m pip install -r requirements.txt

# Verify PyTorch wasn't overwritten by requirements.txt
TORCH_CUDA_AFTER=$(python -c "import torch; print(torch.version.cuda)")
if [ "$TORCH_CUDA_AFTER" != "12.4" ]; then
    echo "ERROR: PyTorch was overwritten by requirements.txt! Found CUDA $TORCH_CUDA_AFTER instead of 12.4"
    echo "Reinstalling correct PyTorch version..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
    echo "✓ PyTorch reinstalled with CUDA 12.4"
fi

# Clone source repository of FoundationPose
# git clone https://github.com/NVlabs/FoundationPose.git

# pip install gdown

git clone https://github.com/wkentaro/gdown
sed -i 's/MAX_NUMBER_FILES = 50/MAX_NUMBER_FILES = 10000/' gdown/gdown/download_folder.py
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
echo "Building nvdiffrast with CUDA 12.8..."
cd ${PROJ_ROOT}/nvdiffrast && pip install . --no-build-isolation

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(($(nproc)-1))

# Install mycuda
echo "Building mycuda..."
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python3 -m pip install -e . --no-build-isolation

cd ${PROJ_ROOT}
