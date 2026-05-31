#!/bin/bash
#SBATCH --job-name=facerec-adaface
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ===========================================================================
# Train IResNet-100 + AdaFace on Leonardo (4 nodes, 16× A100, InfiniBand).
#
# One SLURM task per node owns all 4 local GPUs; torchrun spawns 4 worker
# processes per node (16 total). Nodes coordinate via c10d rendezvous.
# Effective batch = 128 × 16 = 2048 — adjust --lr accordingly (linear scaling
# from the 512-batch baseline: lr = 0.1 × 2048/512 = 0.4).
#
#   sbatch scripts/job_submit.sh
# ===========================================================================

set -euo pipefail

# --- clean environment so nothing from the login shell leaks in -----------
unset PYTHONPATH LD_PRELOAD
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
unset CUDA_VISIBLE_DEVICES NCCL_SOCKET_IFNAME

# --- paths -----------------------------------------------------------------
export PIXI_CACHE_DIR="$CINECA_SCRATCH/.pixi-cache"
mkdir -p "$PIXI_CACHE_DIR"

PROJECT_DIR="$CINECA_SCRATCH/facial-recognition"
DATA_ROOT="$CINECA_SCRATCH/datasets/glint360k-wds"
OUTPUT_DIR="$CINECA_SCRATCH/facial-recognition/checkpoints/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

# --- proxy for low-bandwidth traffic (compute nodes have no internet) -------
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# --- runtime tuning ---------------------------------------------------------
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 4 ))
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================================================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Host        : $(hostname)"
echo "Node list   : ${SLURM_JOB_NODELIST}"
echo "Data root   : ${DATA_ROOT}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "OMP threads : ${OMP_NUM_THREADS}"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "================================================================"

# --- install GPU environment on the compute node ----------------------------
echo "[pixi] Installing GPU environment..."
CONDA_OVERRIDE_CUDA=12.0 pixi install -e gpu
echo "[pixi] Done."

# --- multi-node rendezvous ---------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 1000 ))  # avoid port collisions

echo "[torchrun] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[torchrun] nnodes=$SLURM_NNODES  nproc_per_node=4  world_size=$(( SLURM_NNODES * 4 ))"

# --- launch training ---------------------------------------------------------
echo "[train] Starting..."
pixi run -e gpu python -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 4 \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    src/train_leonardo.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 128 \
    --epochs 20 \
    --lr 0.4 \
    --warmup-epochs 1 \
    --amp-dtype bf16 \
    --num-workers -1

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Job finished with exit code: $EXIT_CODE"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "================================================================"
exit $EXIT_CODE
