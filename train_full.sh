#!/usr/bin/env bash
set -uo pipefail   # NOT -e: we handle the failure branch ourselves (stay-up-to-debug)

# ============================================================
#  CONFIG — edit these, nothing below needs touching
# ============================================================

# --- run mode -----------------------------------------------
TEST_RUN=false          # true = quick smoke (few iters), false = full run
SHUTDOWN=on_success     # off | on_success | always
#   off         -> never shut down (box stays up)
#   on_success  -> shut down only if training exits 0
#   always      -> shut down even if training failed (use with care)

# --- training hyperparameters (full run) --------------------
GAMMA=2                 # R1 gamma (chosen by listening)
BATCH=16                # measured optimum on L4
ITER=320001            # 5000 kimg at batch 16
CKPT_EVERY=10000
SAMPLE_EVERY=10000
AUGMENT=true           # ADA inert at 100k clips; leave off

# --- test-run overrides (used only when TEST_RUN=true) ------
TEST_ITER=11
TEST_CKPT_EVERY=10
TEST_SAMPLE_EVERY=10

# --- paths / env --------------------------------------------
REPO=/home/ubuntu/repos/stylegan2-pytorch
DATA=/scratch/npy_audiomnist_aug
ARCH="8.9"              # L4 = sm_89   (use 12.0 on a Blackwell box)

# ============================================================
#  BODY — leave alone
# ============================================================

cd "$REPO"

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# pick effective params based on TEST_RUN
if [ "$TEST_RUN" = true ]; then
  EFF_ITER=$TEST_ITER
  EFF_CKPT=$TEST_CKPT_EVERY
  EFF_SAMPLE=$TEST_SAMPLE_EVERY
  RUN="$REPO/runs/test_g${GAMMA}"
else
  EFF_ITER=$ITER
  EFF_CKPT=$CKPT_EVERY
  EFF_SAMPLE=$SAMPLE_EVERY
  RUN="$REPO/runs/full_g${GAMMA}"
fi

LOG="$RUN/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RUN/sample" "$RUN/checkpoint"

# optional --augment flag
AUG_FLAG=""
if [ "$AUGMENT" = true ]; then
  AUG_FLAG="--augment --augment_mode audio"
fi

# guard: dataset must be staged (/scratch wipes on stop)
if [ ! -d "$DATA" ]; then
  echo "ERROR: $DATA missing — re-stage the dataset before launching." | tee -a "$LOG"
  exit 1
fi

echo "=== TEST_RUN=$TEST_RUN gamma=$GAMMA batch=$BATCH iter=$EFF_ITER augment=$AUGMENT shutdown=$SHUTDOWN start $(date) ===" | tee -a "$LOG"

( cd "$RUN" && python "$REPO/train.py" \
    --size 128 --batch "$BATCH" --img_channels 1 --dataset npy --seed 0 \
    --iter "$EFF_ITER" --r1 "$GAMMA" \
    --ckpt_every "$EFF_CKPT" --sample_every "$EFF_SAMPLE" \
    $AUG_FLAG \
    "$DATA" ) 2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}   # python's real exit code, not tee's

if [ "$STATUS" -eq 0 ]; then
  echo "=== training OK (exit 0) $(date) ===" | tee -a "$LOG"
else
  echo "=== training FAILED (exit $STATUS) $(date) ===" | tee -a "$LOG"
fi

# SHUTDOWN=off
# shutdown decision
case "$SHUTDOWN" in
  off)
    echo "=== shutdown=off; box staying up ===" | tee -a "$LOG"
    ;;
  on_success)
    if [ "$STATUS" -eq 0 ]; then
      echo "=== shutdown=on_success and run OK; powering off ===" | tee -a "$LOG"
      sudo shutdown -h now
    else
      echo "=== shutdown=on_success but run FAILED; staying up to debug ===" | tee -a "$LOG"
    fi
    ;;
  always)
    echo "=== shutdown=always; powering off regardless of exit $STATUS ===" | tee -a "$LOG"
    sudo shutdown -h now
    ;;
  *)
    echo "=== unknown SHUTDOWN='$SHUTDOWN'; staying up ===" | tee -a "$LOG"
    ;;
esac

exit "$STATUS"