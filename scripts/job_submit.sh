#!/bin/bash
#SBATCH --job-name=facerec-adaface
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ===========================================================================
# Train IResNet-100 + AdaFace on Leonardo (1 node, 4x A100, NVLink).
#
# One SLURM task owns all 4 GPUs; torchrun spawns 4 worker processes (one per
# GPU) inside it. The 24h wall-clock limit means a single run may not finish --
# the training script auto-resumes from <output-dir>/latest.pth, so simply
# re-submitting this script continues where the previous job stopped.
#
#   sbatch scripts/job_submit.sh
# ===========================================================================

set -euo pipefail

# --- clean the environment so nothing from the login shell leaks in ---------
module purge 2>/dev/null || true
unset PYTHONPATH LD_PRELOAD
# Let torchrun --standalone set these itself; stale values break rendezvous.
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
unset CUDA_VISIBLE_DEVICES NCCL_SOCKET_IFNAME

# --- paths (everything lives on $SCRATCH; $HOME only holds the pixi binary) --
export PIXI_CACHE_DIR="${CINECA_SCRATCH:-$SCRATCH}/.pixi-cache"
PROJECT_DIR="$SCRATCH/facial-recognition"
DATA_ROOT="$SCRATCH/datasets/glint360k"
OUTPUT_DIR="$SCRATCH/facial-recognition/checkpoints/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Locate pixi: prefer $PIXI_BIN, then the standard install path, then PATH.
PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
if [[ ! -x "$PIXI_BIN" ]]; then
    PIXI_BIN="$(command -v pixi)"
fi

cd "$PROJECT_DIR"

# --- runtime tuning ---------------------------------------------------------
# Split the 32 CPUs evenly across the 4 GPU processes for math threads.
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 4 ))
# Surface NCCL collective errors instead of hanging (new + legacy var names).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Compute nodes have no internet; make sure no library tries to phone home.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================================================================"
echo "host        : $(hostname)"
echo "job id      : ${SLURM_JOB_ID}"
echo "data root   : ${DATA_ROOT}"
echo "output dir  : ${OUTPUT_DIR}"
echo "pixi        : ${PIXI_BIN}"
echo "OMP threads : ${OMP_NUM_THREADS}"
nvidia-smi || true
echo "================================================================"

# --- launch -----------------------------------------------------------------
"$PIXI_BIN" run --manifest-path "$PROJECT_DIR/pixi.toml" -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train_leonardo.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 128 \
    --epochs 20 \
    --lr 0.1 \
    --warmup-epochs 1 \
    --amp-dtype bf16 \
    --num-workers -1
