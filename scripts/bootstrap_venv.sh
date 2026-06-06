#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/venv"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
INSTALL_DALI="${INSTALL_DALI:-0}"
DALI_PKG="${DALI_PKG:-nvidia-dali-cuda120}"

python3 -m venv --without-pip "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

mkdir -p "${PROJECT_ROOT}/work"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${PROJECT_ROOT}/work/get-pip.py"
python "${PROJECT_ROOT}/work/get-pip.py"

python -m pip install --upgrade pip wheel setuptools

# Install torch/torchvision from the official PyTorch CUDA wheel index.
python -m pip install \
	--index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
	"torch==${TORCH_VERSION}" \
	"torchvision==${TORCHVISION_VERSION}"

python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [ "${INSTALL_DALI}" = "1" ]; then
	echo "Installing DALI package: ${DALI_PKG}"
	python -m pip install "${DALI_PKG}"
fi

echo "Applying MXNet NumPy 2.0 compatibility patches..."
python "${PROJECT_ROOT}/scripts/fix_mxnet_numpy2_compat.py"

cat <<EOF
Environment ready:
- Venv: ${VENV_PATH}
- Activate: source ${VENV_PATH}/bin/activate
- Torch build: torch==${TORCH_VERSION}, torchvision==${TORCHVISION_VERSION}, index=${TORCH_CUDA_TAG}
- DALI: install with INSTALL_DALI=1 (package: ${DALI_PKG})
EOF
