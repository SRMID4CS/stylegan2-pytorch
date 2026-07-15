#!/usr/bin/env bash

set -euo pipefail

REPO=/home/ubuntu/repos/stylegan2-pytorch

DATA=/scratch/npy_audiomnist_aug

cd "$REPO"

#export CUDA_HOME="$CONDA_PREFIX"

#export TORCH_CUDA_ARCH_LIST="8.9"

#export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"

#export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

run_gamma () {

  local G=$1

  local DIR="$REPO/runs/g${G}"

  mkdir -p "$DIR/sample" "$DIR/checkpoint"

  echo "=== R1 gamma=${G} -> ${DIR} ==="

  ( cd "$DIR" && python "$REPO/train.py" \

      --size 128 --batch 16 --img_channels 1 --dataset npy --seed 0 \

      --augment --augment_mode audio \

      --iter 7001 --r1 "${G}" \

      --ckpt_every 1000 --sample_every 1000 \

      "$DATA" ) 2>&1 | tee "$DIR/train.log"

}

run_gamma 10

run_gamma 20

echo "=== sweep complete; shutting down ==="

sudo shutdown -h now