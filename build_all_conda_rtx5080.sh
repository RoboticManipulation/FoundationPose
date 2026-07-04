#!/usr/bin/env bash

# Reproducible FoundationPose + ROS 2 setup for NVIDIA RTX 5080 (sm_120).
#
# Run from anywhere inside the FoundationPoseROS2 devcontainer:
#   bash build_all_conda_rtx5080.sh
#
# Optional controls:
#   CONDA_ENV_NAME=foundationpose_ros  Conda environment to use.
#   BUILD_JOBS=8                       Parallel compiler jobs.
#   BUILD_ROS_BRIDGE=0                 Skip the ROS 2 bridge build.
#   RUN_INFERENCE_SMOKE_TEST=0         Skip the bundled one-frame inference.

set -Eeuo pipefail

trap 'printf "ERROR: line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-foundationpose_ros}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
BUILD_JOBS="${BUILD_JOBS:-8}"
BUILD_ROS_BRIDGE="${BUILD_ROS_BRIDGE:-1}"
RUN_INFERENCE_SMOKE_TEST="${RUN_INFERENCE_SMOKE_TEST:-1}"

PYTORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
PYTORCH3D_REVISION="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba" # v0.7.9
NVDIFFRAST_REVISION="253ac4fcea7de5f396371124af597e6cc957bfae" # v0.4.0
GDOWN_REVISION="a59d00d73056c2b282787ed711d4eb5c1cefa504"
EIGEN_VERSION="3.4.0"
EIGEN_SHA256="8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS="${BUILD_JOBS}"

log() {
  printf '\n==> %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

activate_conda() {
  local conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"

  if [[ ! -f "${conda_sh}" ]]; then
    printf 'Conda initialization script not found: %s\n' "${conda_sh}" >&2
    exit 1
  fi

  # Conda and ROS setup scripts reference variables that may initially be unset.
  set +u
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV_NAME}"
  set -u
}

checkout_pinned_repo() {
  local url="$1"
  local destination="$2"
  local revision="$3"

  if [[ ! -d "${destination}/.git" ]]; then
    rm -rf "${destination}"
    git clone "${url}" "${destination}"
  fi

  if [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=no)" ]]; then
    printf 'Refusing to overwrite tracked changes in %s\n' "${destination}" >&2
    exit 1
  fi

  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"

  if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${revision}" ]]; then
    printf 'Failed to check out required revision %s in %s\n' "${revision}" "${destination}" >&2
    exit 1
  fi
}

download_weight() {
  local google_drive_id="$1"
  local relative_path="$2"
  local expected_sha256="$3"
  local destination="${PROJECT_ROOT}/${relative_path}"
  local temporary_file="${destination}.part"

  if [[ -f "${destination}" ]] && \
      printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status; then
    printf 'Verified existing %s\n' "${relative_path}"
    return
  fi

  mkdir -p "$(dirname -- "${destination}")"
  rm -f "${temporary_file}"
  python -m gdown "${google_drive_id}" --output "${temporary_file}"

  if ! printf '%s  %s\n' "${expected_sha256}" "${temporary_file}" | \
      sha256sum --check --status; then
    rm -f "${temporary_file}"
    printf 'Checksum verification failed for %s\n' "${relative_path}" >&2
    exit 1
  fi

  mv "${temporary_file}" "${destination}"
  printf 'Downloaded and verified %s\n' "${relative_path}"
}

install_eigen() {
  local work_dir
  local archive
  work_dir="$(mktemp -d)"
  archive="${work_dir}/eigen-${EIGEN_VERSION}.tar.gz"
  trap 'rm -rf "${work_dir}"' RETURN

  curl -fsSL \
    "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" \
    -o "${archive}"
  printf '%s  %s\n' "${EIGEN_SHA256}" "${archive}" | sha256sum --check --status
  tar -xzf "${archive}" -C "${work_dir}"
  cmake \
    -S "${work_dir}/eigen-${EIGEN_VERSION}" \
    -B "${work_dir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTING=OFF
  sudo cmake --install "${work_dir}/build"

  rm -rf "${work_dir}"
  trap - RETURN
}

build_mycpp() {
  local pybind11_cmake_dir
  pybind11_cmake_dir="$(python -m pybind11 --cmakedir)"

  rm -rf "${PROJECT_ROOT}/mycpp/build"
  cmake \
    -S "${PROJECT_ROOT}/mycpp" \
    -B "${PROJECT_ROOT}/mycpp/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${pybind11_cmake_dir};/usr/local/share/eigen3/cmake"
  cmake --build "${PROJECT_ROOT}/mycpp/build" --parallel "${BUILD_JOBS}"
}

build_ros_bridge() {
  local ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  local install_scripts="${PROJECT_ROOT}/ros2_bridge_ws/install/foundationpose_bridge/lib/foundationpose_bridge"

  if [[ ! -f "${ros_setup}" ]]; then
    printf 'ROS setup script not found: %s\n' "${ros_setup}" >&2
    exit 1
  fi

  set +u
  # shellcheck disable=SC1090
  source "${ros_setup}"
  set -u

  (
    cd "${PROJECT_ROOT}/ros2_bridge_ws"
    colcon build --symlink-install
  )

  # Console scripts must use the Conda Python that owns the ML dependencies.
  if [[ -d "${install_scripts}" ]]; then
    while IFS= read -r -d '' script; do
      if head -n 1 "${script}" | grep -q '^#!/usr/bin/python'; then
        sed -i "1s|^#!/usr/bin/python.*|#!${CONDA_PREFIX}/bin/python3|" "${script}"
      fi
    done < <(find "${install_scripts}" -maxdepth 1 -type f -perm /111 -print0)
  fi
}

run_cuda_smoke_tests() {
  python - <<'PY'
import importlib
import numpy
import torch

assert torch.__version__.startswith("2.7.1+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert numpy.__version__ == "1.26.4", numpy.__version__

value = (torch.arange(8, device="cuda", dtype=torch.float32) * 2).sum().item()
assert value == 56.0, value

from pytorch3d.ops import knn_points
points = torch.rand(1, 16, 3, device="cuda")
knn_points(points, points, K=2)

import nvdiffrast.torch as dr
dr.RasterizeCudaContext()

importlib.import_module("mycpp.build.mycpp")
importlib.import_module("bundlesdf.mycuda.common")
importlib.import_module("estimater")

print(
    "CUDA smoke tests passed:",
    torch.cuda.get_device_name(0),
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
)
PY
}

run_inference_smoke_test() {
  local demo_root="${PROJECT_ROOT}/demo_data/mustard0"

  if [[ ! -f "${demo_root}/mesh/textured_simple.obj" ]]; then
    printf 'Skipping inference smoke test: bundled mustard0 demo data is absent.\n'
    return
  fi

  (
    cd "${PROJECT_ROOT}"
    python - <<'PY'
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed
from datareader import YcbineoatReader
import numpy as np
import os
import trimesh

set_seed(0)
root = os.getcwd()
demo_root = os.path.join(root, "demo_data", "mustard0")
mesh = trimesh.load(os.path.join(demo_root, "mesh", "textured_simple.obj"))
estimator = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=ScorePredictor(),
    refiner=PoseRefinePredictor(),
    debug_dir="/tmp/foundationpose_rtx5080_smoke",
    debug=0,
    glctx=dr.RasterizeCudaContext(),
)
reader = YcbineoatReader(video_dir=demo_root, shorter_side=None, zfar=np.inf)
pose = estimator.register(
    K=reader.K,
    rgb=reader.get_color(0),
    depth=reader.get_depth(0),
    ob_mask=reader.get_mask(0).astype(bool),
    iteration=1,
)
assert pose.shape == (4, 4), pose.shape
assert np.isfinite(pose).all(), pose
print("FoundationPose one-frame inference smoke test passed.")
PY
  )
}

log "Preflight"
activate_conda
require_command curl
require_command git
require_command cmake
require_command sha256sum
require_command sudo
require_command "${CUDA_HOME}/bin/nvcc"
sudo -n true

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  printf 'The %s environment must use Python 3.10.\n' "${CONDA_ENV_NAME}" >&2
  exit 1
fi

"${CUDA_HOME}/bin/nvcc" --version | tail -n 1

log "Pinned Python packaging tools"
python -m pip install \
  pip==25.1.1 \
  setuptools==75.8.0 \
  wheel==0.45.1

log "PyTorch ${PYTORCH_VERSION} with CUDA 12.8 and Blackwell support"
python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

# visdom 0.2.4 still imports pkg_resources while building. Installing it with
# the pinned setuptools and without build isolation avoids setuptools 82+.
log "FoundationPose Python dependencies"
python -m pip install "visdom==0.2.4" --no-build-isolation
python -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"
python -m pip install \
  "empy==3.3.4" \
  "catkin_pkg==1.1.0" \
  "gdown @ git+https://github.com/wkentaro/gdown.git@${GDOWN_REVISION}"

log "Verified FoundationPose model weights"
download_weight \
  "1chawjJVATReUoWtX7v1euKe2tMNSx_F-" \
  "weights/2023-10-28-18-33-37/config.yml" \
  "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686"
download_weight \
  "1mOurS4MDYbnL7Y8jAOQNqgs6N-KEeczG" \
  "weights/2023-10-28-18-33-37/model_best.pth" \
  "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60"
download_weight \
  "1Nu2edRUomCWNs-2DPTxTYFE9ZsuG80Yk" \
  "weights/2024-01-11-20-02-45/config.yml" \
  "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f"
download_weight \
  "1L6Iv7F8sS0MQmzBpWfo_4UxNv6_KCRhe" \
  "weights/2024-01-11-20-02-45/model_best.pth" \
  "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26"

log "Eigen ${EIGEN_VERSION} and mycpp"
install_eigen
build_mycpp

log "PyTorch3D v0.7.9 for CUDA 12.8 / sm_120"
python -m pip install \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REVISION}"

log "nvdiffrast v0.4.0 for CUDA 12.8 / sm_120"
checkout_pinned_repo \
  "https://github.com/NVlabs/nvdiffrast.git" \
  "${PROJECT_ROOT}/nvdiffrast" \
  "${NVDIFFRAST_REVISION}"
python -m pip install \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  "${PROJECT_ROOT}/nvdiffrast"

log "FoundationPose mycuda for CUDA 12.8 / sm_120"
rm -rf \
  "${PROJECT_ROOT}/bundlesdf/mycuda/build" \
  "${PROJECT_ROOT}/bundlesdf/mycuda"/*.egg-info \
  "${PROJECT_ROOT}/bundlesdf/mycuda"/*.so
python -m pip install \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  --editable "${PROJECT_ROOT}/bundlesdf/mycuda"

log "CUDA and native-extension smoke tests"
run_cuda_smoke_tests

if [[ "${RUN_INFERENCE_SMOKE_TEST}" == "1" ]]; then
  log "One-frame FoundationPose inference smoke test"
  run_inference_smoke_test
fi

if [[ "${BUILD_ROS_BRIDGE}" == "1" ]]; then
  log "ROS 2 FoundationPose bridge"
  build_ros_bridge
fi

log "RTX 5080 setup completed successfully"
printf 'Activate with: conda activate %s\n' "${CONDA_ENV_NAME}"
if [[ "${BUILD_ROS_BRIDGE}" == "1" ]]; then
  printf 'ROS setup: source %s/ros2_bridge_ws/install/setup.bash\n' "${PROJECT_ROOT}"
fi
