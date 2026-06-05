#!/usr/bin/env bash
set -euo pipefail

# Robust Google Drive large-file downloader using byte-range requests.
# This avoids the HTML interstitial behavior seen on plain GET requests.

FILE_ID="1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR"
OUT_PATH="${1:-data/archives/ms1m_from_drive.archive}"
BASE_URL="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download"
CHUNK_BYTES="${CHUNK_BYTES:-134217728}"

get_uuid() {
  curl -sSL "${BASE_URL}" \
    | sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' \
    | head -n1
}

get_total_bytes() {
  local uuid="$1"
  curl -sSI -H 'Range: bytes=0-0' "${BASE_URL}&confirm=t&uuid=${uuid}" \
    | sed -n 's/.*[Cc]ontent-[Rr]ange: bytes 0-0\/\([0-9][0-9]*\).*/\1/p' \
    | tail -n1
}

mkdir -p "$(dirname "${OUT_PATH}")"

if [[ ! -f "${OUT_PATH}" ]]; then
  : > "${OUT_PATH}"
fi

uuid="$(get_uuid)"
total_bytes="$(get_total_bytes "${uuid}")"
if [[ -z "${total_bytes}" ]]; then
  echo "ERROR: Could not detect total file size from Drive response" >&2
  exit 1
fi

echo "Expected bytes: ${total_bytes}"
echo "Chunk bytes: ${CHUNK_BYTES}"

while true; do
  current_size="$(stat -c%s "${OUT_PATH}")"
  if [[ "${current_size}" -ge "${total_bytes}" ]]; then
    break
  fi

  end_byte=$((current_size + CHUNK_BYTES - 1))
  if [[ "${end_byte}" -ge "${total_bytes}" ]]; then
    end_byte=$((total_bytes - 1))
  fi
  expected_chunk=$((end_byte - current_size + 1))

  uuid="$(get_uuid)"
  url="${BASE_URL}&confirm=t&uuid=${uuid}"
  echo "Fetching bytes ${current_size}-${end_byte}/${total_bytes} (uuid=${uuid})"

  tmp_chunk="$(mktemp)"
  if ! curl -L --fail --retry 8 --retry-delay 5 -r "${current_size}-${end_byte}" "${url}" -o "${tmp_chunk}"; then
    rm -f "${tmp_chunk}"
    echo "Range request failed, retrying in 5s..." >&2
    sleep 5
    continue
  fi

  actual_chunk="$(stat -c%s "${tmp_chunk}")"
  if [[ "${actual_chunk}" -ne "${expected_chunk}" ]]; then
    rm -f "${tmp_chunk}"
    echo "Unexpected chunk size ${actual_chunk} (expected ${expected_chunk}), retrying..." >&2
    sleep 5
    continue
  fi

  cat "${tmp_chunk}" >> "${OUT_PATH}"
  rm -f "${tmp_chunk}"

  new_size="$(stat -c%s "${OUT_PATH}")"
  echo "Progress: ${new_size}/${total_bytes} bytes"

  if [[ "${new_size}" -gt "${total_bytes}" ]]; then
    echo "ERROR: Download exceeded expected size; refusing to continue" >&2
    exit 1
  fi

  if [[ "${new_size}" -le "${current_size}" ]]; then
    echo "No forward progress, retrying in 5s..." >&2
    sleep 5
  fi
done

echo "Download complete: ${OUT_PATH}"
ls -lh "${OUT_PATH}"
