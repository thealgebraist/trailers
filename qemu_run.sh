#!/usr/bin/env bash
set -euo pipefail

# qemu_run.sh - create and run an Ubuntu 22.04 cloud image VM and provision it with cloud-init
# Requirements on host: qemu-system-x86_64, qemu-img, cloud-localds (cloud-image-utils), wget or curl
# Usage: ./qemu_run.sh [GIT_REPO_URL]

REPO=${1:-https://github.com/thealgebraist/dalek-comes-home.git}
IMG_URL=https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img
BASE_IMG=jammy-server-cloudimg-amd64.img
LOCAL_IMG=ubuntu.qcow2
SEED=seed.img
CLOUD_DIR=cloud-init
SSH_PORT=2222
MEM=8192
CPUS=4

mkdir -p qemu && cd qemu

if [ ! -f "$BASE_IMG" ]; then
  echo "Downloading Ubuntu cloud image..."
  if command -v wget >/dev/null 2>&1; then
    wget -O "$BASE_IMG" "$IMG_URL"
  else
    curl -L -o "$BASE_IMG" "$IMG_URL"
  fi
fi

# Create a writable qcow2 based on the downloaded base image
if [ ! -f "$LOCAL_IMG" ]; then
  echo "Creating overlay image $LOCAL_IMG (20G)..."
  qemu-img create -f qcow2 -b "$BASE_IMG" -F qcow2 "$LOCAL_IMG" 20G
fi

# Prepare cloud-init seed
if [ ! -d "$CLOUD_DIR" ]; then
  cat > user-data <<'EOF'
#cloud-config
users:
  - name: ubuntu
    gecos: Ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: users, admin
    lock_passwd: false
    passwd: $6$rounds=4096$uI8b...$EXAMPLECHANGEIT
    ssh_pwauth: True

chpasswd:
  list: |
    ubuntu:ubuntu
  expire: False

ssh_pwauth: True

packages:
  - git
  - python3-venv
  - python3-pip
  - build-essential
  - clang
  - cmake
  - cloud-image-utils
  - qemu-guest-agent

runcmd:
  - [ bash, -lc, "set -x" ]
  - [ bash, -lc, "git clone ${REPO} /home/ubuntu/dalek-comes-home || (cd /home/ubuntu/dalek-comes-home && git pull)" ]
  - [ bash, -lc, "cd /home/ubuntu/dalek-comes-home && ./setup.sh || true" ]
  - [ bash, -lc, "cd /home/ubuntu/dalek-comes-home && make export-tokenizers build-cpp || true" ]
  - [ bash, -lc, "touch /home/ubuntu/provision_done" ]
  - [ bash, -lc, "poweroff -f" ]
EOF

  cat > meta-data <<'EOF'
instance-id: iid-local01
local-hostname: ubuntu-qemu
EOF

  mkdir -p "$CLOUD_DIR"
  mv user-data meta-data "$CLOUD_DIR/"
fi

# Create seed ISO
if [ ! -f "$SEED" ]; then
  echo "Creating cloud-init seed ISO..."
  if command -v cloud-localds >/dev/null 2>&1; then
    cloud-localds "$SEED" "$CLOUD_DIR/user-data" --network-config=\"\" || cloud-localds "$SEED" "$CLOUD_DIR/user-data"
  else
    # Fallback: use mkisofs/genisoimage or hdiutil (macOS)
    if command -v mkisofs >/dev/null 2>&1; then
      mkisofs -o "$SEED" -V cidata -J -r "$CLOUD_DIR/user-data" "$CLOUD_DIR/meta-data"
    elif command -v genisoimage >/dev/null 2>&1; then
      genisoimage -o "$SEED" -V cidata -J -r "$CLOUD_DIR/user-data" "$CLOUD_DIR/meta-data"
    elif [[ "$(uname)" == "Darwin" ]] && command -v hdiutil >/dev/null 2>&1; then
      # hdiutil on macOS tends to add .iso; create a temporary file and normalize
      tmp_iso="$SEED.iso"
      hdiutil makehybrid -o "$tmp_iso" "$CLOUD_DIR" -iso -joliet
      if [ -f "$tmp_iso" ]; then
        mv "$tmp_iso" "$SEED"
      fi
    else
      echo "cloud-localds/mkisofs/genisoimage/hdiutil not found; cannot create seed ISO" >&2
      exit 1
    fi
  fi
fi

# Run QEMU with forwarded SSH (host port $SSH_PORT -> guest 22)
echo "Launching QEMU; SSH to the VM after provisioning as: ssh ubuntu@localhost -p $SSH_PORT (password: ubuntu)"
qemu-system-x86_64 \
  -machine accel=tcg -m "$MEM" -smp "$CPUS" \
  -drive file="$LOCAL_IMG",if=virtio,format=qcow2 \
  -drive file="$SEED",if=virtio,format=raw \
  -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 -device virtio-net-pci,netdev=net0 \
  -nographic
