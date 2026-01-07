#!/usr/bin/env bash
set -euo pipefail

# Build x86_64 CPU (and optionally CUDA) binaries inside an amd64 container using QEMU emulation.
# Usage: ./scripts/build_x86_binaries.sh [cpu|cuda]
MODE=${1:-cpu}
ROOT=$(pwd)
IMAGE=ubuntu:22.04

echo "Building x86_64 binaries (mode=$MODE) inside image $IMAGE"

# Choose container runtime
if command -v /usr/local/bin/container >/dev/null 2>&1; then
  CTR='/usr/local/bin/container run --rm'
elif command -v docker >/dev/null 2>&1; then
  CTR='docker run --rm --platform linux/amd64'
elif command -v podman >/dev/null 2>&1; then
  CTR='podman run --rm --platform linux/amd64'
else
  echo "No container runtime found (container/docker/podman). Install one or use the QEMU VM." >&2
  exit 1
fi

# Run build inside container
$CTR -v "$ROOT":/workspace -w /workspace -e DEBIAN_FRONTEND=noninteractive $IMAGE bash -lc "set -eux; apt-get update; apt-get install -y build-essential cmake git wget curl pkg-config ca-certificates || true; cd /workspace; ./setup.sh || true; mkdir -p build && cd build; cmake .. -DCMAKE_BUILD_TYPE=Release || true; make -j\$(nproc) || true; cd /workspace; if [ \"$MODE\" = cuda ]; then echo 'CUDA build requested: please run a CUDA-enabled image or VM with NVIDIA passthrough and build there.'; fi; echo 'Build finished; copy artifacts to /workspace/bin (if produced).'"

echo "Done. If the container produced x86_64 binaries, they will be in ./bin with appropriate names."
