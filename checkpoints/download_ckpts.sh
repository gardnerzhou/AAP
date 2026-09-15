#!/bin/bash
set -e

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/sam2_hiera_large.pt"

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget -O "$OUT" "$sam2_hiera_l_url"
echo "Checkpoint saved to $OUT"
