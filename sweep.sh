#!/usr/bin/env bash
#
# R1 (gamma) sweep — short runs at 2-3 --r1 values, then pick by the mel-native
# convergence curve (eval/convergence_curve.py) and train long with train_full.sh.
#
# Dataset-agnostic: everything below is an env-overridable knob, so the SAME
# sweep runs on AudioMNIST and Speech Commands (SPEECH_COMMANDS_SPEC.md §7).
# Defaults reproduce the original AudioMNIST sweep.
#
#   # AudioMNIST (defaults)
#   ./sweep.sh
#
#   # Speech Commands: ~97k clips, augmentation OFF (SC spec §7), 35 word classes
#   DATA=/scratch/npy_speech_commands DATASET=speech_commands AUGMENT=false \
#   GAMMAS="10 20" ./sweep.sh
#
# Note the R1 gamma values are a sweep knob, not a contract value — GAMMAS is the
# list to try. Everything the eval needs (speaker split, word labels, class count)
# is read from the prepared --npy-dir, so nothing here is dataset-specific.

set -euo pipefail

REPO="${REPO:-/home/ubuntu/repos/stylegan2-pytorch}"

# --- what to sweep, on what ---------------------------------------------------
DATA="${DATA:-/scratch/npy_audiomnist_aug}"   # prepare_audio_data.py --out dir
DATASET="${DATASET:-audiomnist}"              # eval cache namespace + label rule
GAMMAS="${GAMMAS:-2 10 20}"                     # --r1 values, space separated
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"

# --- training knobs -----------------------------------------------------------
BATCH="${BATCH:-16}"
ITER="${ITER:-7001}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
SEED="${SEED:-0}"
AUGMENT="${AUGMENT:-true}"                    # SC: set false (~97k clips, spec §7)
ARCH_LIST="${ARCH_LIST:-}"                    # e.g. 8.9 (L4/L40S) or 12.0 (Blackwell)

# --- convergence eval after each run (how the winner is picked) ---------------
RUN_EVAL="${RUN_EVAL:-true}"
EVAL_LABEL_MODE="${EVAL_LABEL_MODE:-both}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-2000}"      # FROZEN across the whole sweep
EVAL_SEED="${EVAL_SEED:-0}"                   # FROZEN across the whole sweep

SHUTDOWN="${SHUTDOWN:-true}"                  # power off when the sweep completes

cd "$REPO"

if [ -n "$ARCH_LIST" ]; then
  export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
fi

if [ ! -d "$DATA" ]; then
  echo "ERROR: $DATA missing — re-stage the prepared .npy dir before launching." >&2
  exit 1
fi

AUG_FLAG=""
if [ "$AUGMENT" = true ]; then
  AUG_FLAG="--augment --augment_mode audio"
fi

run_gamma () {
  local G=$1
  local DIR="$RUNS_DIR/g${G}"

  mkdir -p "$DIR/sample" "$DIR/checkpoint"
  echo "=== R1 gamma=${G} data=${DATA} augment=${AUGMENT} -> ${DIR} ==="

  ( cd "$DIR" && python "$REPO/train.py" \
      --size 128 --batch "$BATCH" --img_channels 1 --dataset npy --seed "$SEED" \
      $AUG_FLAG \
      --iter "$ITER" --r1 "${G}" \
      --ckpt_every "$CKPT_EVERY" --sample_every "$SAMPLE_EVERY" \
      "$DATA" ) 2>&1 | tee "$DIR/train.log"

  if [ "$RUN_EVAL" = true ]; then
    echo "=== convergence curve for gamma=${G} ==="
    # --seed/--n-samples are frozen across the sweep: the absolute FD is only
    # comparable between runs when the reference, seed and N are identical.
    python "$REPO/eval/convergence_curve.py" \
      --npy-dir "$DATA" --ckpt-glob "$DIR/checkpoint/*.pt" \
      --dataset "$DATASET" --label-mode "$EVAL_LABEL_MODE" \
      --n-samples "$EVAL_N_SAMPLES" --seed "$EVAL_SEED" \
      --out "$DIR/convergence_curve" 2>&1 | tee "$DIR/eval.log" \
      || echo "WARNING: convergence eval failed for gamma=${G} (training output kept)"
  fi
}

for G in $GAMMAS; do
  run_gamma "$G"
done

echo "=== sweep complete (gammas: $GAMMAS) ==="
if [ "$SHUTDOWN" = true ]; then
  echo "=== shutting down ==="
  sudo shutdown -h now
fi
