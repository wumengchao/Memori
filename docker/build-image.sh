#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-saa-memori}"
TAG="${TAG:-latest}"

echo "=========================================="
echo "Building ${IMAGE_NAME}:${TAG}"
echo "=========================================="

docker build -t "${IMAGE_NAME}:${TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${ROOT_DIR}"

echo ""
echo "✅ Build complete: ${IMAGE_NAME}:${TAG}"
docker images "${IMAGE_NAME}:${TAG}"
