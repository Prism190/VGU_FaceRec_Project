#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${PROJECT_ROOT}/data/raw"
ARCHIVE_ROOT="${PROJECT_ROOT}/data/archives"
WEIGHT_ROOT="${PROJECT_ROOT}/checkpoints/pretrained"
VENV_PY="${PROJECT_ROOT}/venv/bin/python"
KAGGLE_CLI="${PROJECT_ROOT}/venv/bin/kaggle"

mkdir -p "${RAW_ROOT}" "${ARCHIVE_ROOT}" "${WEIGHT_ROOT}"

adopt_partial_for_resume() {
  local target_path="$1"
  if [ -f "${target_path}" ]; then
    return 0
  fi

  local partial
  partial="$(ls -1S "${target_path}.part"* 2>/dev/null | head -n 1 || true)"
  if [ -n "${partial}" ] && [ -f "${partial}" ]; then
    echo "[RESUME] adopting partial file ${partial} -> ${target_path}"
    mv "${partial}" "${target_path}"
  fi
}

if [ ! -x "${VENV_PY}" ]; then
  echo "Missing venv python at ${VENV_PY}. Run scripts/bootstrap_venv.sh first."
  exit 1
fi

if ! "${VENV_PY}" -c "import gdown" >/dev/null 2>&1; then
  echo "gdown is not installed in venv. Installing..."
  "${VENV_PY}" -m pip install gdown >/dev/null
fi

download_gdrive() {
  local file_id="$1"
  local out_path="$2"
  local label="$3"

  mkdir -p "$(dirname "${out_path}")"
  adopt_partial_for_resume "${out_path}"
  echo "[DL/RESUME] ${label} -> ${out_path}"
  "${VENV_PY}" -m gdown --continue "https://drive.google.com/uc?id=${file_id}" -O "${out_path}"
}

download_http() {
  local url="$1"
  local out_path="$2"
  local label="$3"

  mkdir -p "$(dirname "${out_path}")"
  adopt_partial_for_resume "${out_path}"
  echo "[DL/RESUME] ${label} -> ${out_path}"
  wget -c "${url}" -O "${out_path}"
}

extract_archive() {
  local archive_path="$1"
  local dst_dir="$2"
  local label="$3"

  if [ ! -f "${archive_path}" ]; then
    echo "[WARN] ${label} archive missing: ${archive_path}"
    return 0
  fi

  mkdir -p "${dst_dir}"

  if [ -n "$(find "${dst_dir}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1 || true)" ]; then
    echo "[SKIP] ${label} already extracted: ${dst_dir}"
    return 0
  fi

  if unzip -tqq "${archive_path}" >/dev/null 2>&1; then
    echo "[UNZIP] ${label} -> ${dst_dir}"
    unzip -n "${archive_path}" -d "${dst_dir}" >/dev/null
    return 0
  fi

  if tar -tf "${archive_path}" >/dev/null 2>&1; then
    echo "[UNTAR] ${label} -> ${dst_dir}"
    tar -xf "${archive_path}" -C "${dst_dir}"
    return 0
  fi

  echo "[WARN] ${label}: unknown archive format for ${archive_path}"
}

download_kaggle_dataset() {
  local slug="$1"
  local zip_name="$2"
  local extract_dir="$3"
  local label="$4"

  if [ ! -x "${KAGGLE_CLI}" ]; then
    echo "[SKIP] ${label}: Kaggle CLI not found at ${KAGGLE_CLI}"
    return 0
  fi

  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "[SKIP] ${label}: missing Kaggle token at $HOME/.kaggle/kaggle.json"
    return 0
  fi

  mkdir -p "${ARCHIVE_ROOT}/kaggle"
  local out_zip="${ARCHIVE_ROOT}/kaggle/${zip_name}"

  if [ ! -f "${out_zip}" ]; then
    echo "[DL] ${label} (Kaggle: ${slug})"
    "${KAGGLE_CLI}" datasets download -d "${slug}" -p "${ARCHIVE_ROOT}/kaggle"
    local newest
    newest="$(ls -1t "${ARCHIVE_ROOT}/kaggle"/*.zip | head -n 1)"
    mv "${newest}" "${out_zip}"
  else
    echo "[SKIP] ${label} archive exists: ${out_zip}"
  fi

  extract_archive "${out_zip}" "${extract_dir}" "${label}"
}

# ---------------------------
# Datasets
# ---------------------------
CASIA_ID="1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l"
IJB_ID="1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o"
MS1M_ID="${MS1M_ID:-1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR}"
DOWNLOAD_MS1M="${DOWNLOAD_MS1M:-0}"

download_gdrive "${CASIA_ID}" "${ARCHIVE_ROOT}/casia-webface.zip" "CASIA-WebFace"
download_http "http://www.cfpw.io/cfp-dataset.zip" "${ARCHIVE_ROOT}/cfp-dataset.zip" "CFP dataset"
download_gdrive "${IJB_ID}" "${ARCHIVE_ROOT}/ijb.zip" "IJB dataset"

extract_archive "${ARCHIVE_ROOT}/casia-webface.zip" "${RAW_ROOT}/casia-webface" "CASIA-WebFace"
extract_archive "${ARCHIVE_ROOT}/cfp-dataset.zip" "${RAW_ROOT}/cfp" "CFP dataset"
extract_archive "${ARCHIVE_ROOT}/ijb.zip" "${RAW_ROOT}/ijb" "IJB dataset"

if [ "${DOWNLOAD_MS1M}" = "1" ]; then
  if download_gdrive "${MS1M_ID}" "${ARCHIVE_ROOT}/ms1m_from_drive.archive" "MS1M dataset"; then
    extract_archive "${ARCHIVE_ROOT}/ms1m_from_drive.archive" "${RAW_ROOT}/ms1m" "MS1M dataset"
  else
    echo "[WARN] MS1M download failed (likely Google Drive quota / permission)."
    echo "       Retry later or provide a mirror link."
  fi
fi

# Kaggle-authenticated datasets
# LFW
# https://www.kaggle.com/datasets/atulanandjha/lfwpeople?select=pairs.txt
download_kaggle_dataset \
  "atulanandjha/lfwpeople" \
  "lfwpeople.zip" \
  "${RAW_ROOT}/lfw" \
  "LFW"

# AgeDB-30 package
# https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112
download_kaggle_dataset \
  "yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112" \
  "agedb-30-calfw-cplfw-lfw-aligned-112x112.zip" \
  "${RAW_ROOT}/agedb30_bundle" \
  "AgeDB/CalFW/CPLFW/LFW-aligned bundle"

# ---------------------------
# Teacher model weights
# ---------------------------
# MagFace Model Zoo (from official repository):
# iResNet100 / MS1MV2 / DDP / MagFace
MAGFACE_R100_ID="1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H"
# iResNet18 / CASIA-WebFace / DP / MagFace (lightweight fallback)
MAGFACE_R18_CASIA_ID="18pSIQOHRBQ-srrYfej20S5M8X8b_7zb9"

download_gdrive "${MAGFACE_R100_ID}" "${WEIGHT_ROOT}/magface_iresnet100_ms1mv2.pth" "MagFace iResNet100"
download_gdrive "${MAGFACE_R18_CASIA_ID}" "${WEIGHT_ROOT}/magface_iresnet18_casia.pth" "MagFace iResNet18 CASIA"

# ---------------------------
# LitMAS anti-spoofing weights
# ---------------------------
# DeiT-tiny MoE model from IAB-IITJ/LitMAS (GitHub, 2025)
LITMAS_URL="https://github.com/IAB-IITJ/LitMAS/raw/main/model_weights/downstream_moe_model_litmas.pth"
LITMAS_OUT="${WEIGHT_ROOT}/litmas_downstream_moe.pth"
if [ ! -f "${LITMAS_OUT}" ]; then
    echo "[INFO] Downloading LitMAS downstream MoE model..."
    curl -fL --max-time 120 "${LITMAS_URL}" -o "${LITMAS_OUT}" || \
        echo "[WARN] LitMAS download failed — retry manually: curl -fL '${LITMAS_URL}' -o '${LITMAS_OUT}'"
else
    echo "[SKIP] LitMAS weights already exist: ${LITMAS_OUT}"
fi

# ---------------------------
# Student inference checkpoints (GitHub Releases)
# ---------------------------
RELEASE_BASE="https://github.com/Prism190/AI_FaceRec_VGU_2026/releases/download/v1.0-vgu2026"
STUDENT_ROOT="${PROJECT_ROOT}/runs"

download_student() {
    local PHASE="$1"
    local RUN_DIR="$2"
    local FNAME="mobilenetv4_student_${PHASE}.pt"
    local OUT="${STUDENT_ROOT}/${RUN_DIR}/checkpoints/${FNAME}"
    mkdir -p "$(dirname "${OUT}")"
    if [ ! -f "${OUT}" ]; then
        echo "[INFO] Downloading student checkpoint: ${FNAME} ..."
        curl -fL --max-time 300 "${RELEASE_BASE}/${FNAME}" -o "${OUT}" || \
            echo "[WARN] Download failed — get manually from https://github.com/Prism190/AI_FaceRec_VGU_2026/releases/tag/v1.0-vgu2026"
    else
        echo "[SKIP] Student checkpoint exists: ${OUT}"
    fi
}

download_student "phase1" "ms1m_magface_phase1_cplus_aplus_v1"
download_student "phase2" "ms1m_magface_phase2_occlusion_spatial_v1"
download_student "phase3" "ms1m_magface_phase3_trueasym_swa_v1"

echo ""
echo "Asset download stage complete."
echo "Archives:   ${ARCHIVE_ROOT}"
echo "Raw data:   ${RAW_ROOT}"
echo "Weights:    ${WEIGHT_ROOT}"
