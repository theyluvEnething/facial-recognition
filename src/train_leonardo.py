"""Distributed training entrypoint for the IResNet-100 + AdaFace embedder.

Launched by ``torchrun`` on a single Leonardo node with 4 A100 GPUs::

    torchrun --standalone --nnodes=1 --nproc_per_node=4 src/train_leonardo.py \\
        --data-root $SCRATCH/datasets/glint360k \\
        --output-dir $SCRATCH/facial-recognition/checkpoints/run_X

Each ``torchrun`` worker is one process bound to one GPU (``LOCAL_RANK``). The model
is wrapped in ``DistributedDataParallel``; gradients are all-reduced over NVLink.

Design notes
------------
* **Mixed precision**: bf16 by default (A100 native, no loss scaler needed). fp16 is
  also supported and uses ``torch.amp.GradScaler``. The AdaFace head internally
  forces fp32 for its acos/cos math regardless of the outer autocast dtype.
* **Auto-resume**: Leonardo caps a job at 24h. On startup we look for
  ``<output-dir>/latest.pth`` and resume optimizer + scaler + epoch/step from it, so
  re-submitting the same job script transparently continues training.
* **Checkpoints are written atomically** (write ``.tmp`` then ``os.replace``) so a
  job killed mid-write never corrupts ``latest.pth``.
* **Throughput logging** reports a running loss, the current LR, and images/sec.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Make sibling modules importable whether run as `python src/train_leonardo.py` or
# via torchrun with src/ as the script dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import build_dataloader, build_wds_dataloader  # noqa: E402
from model import FaceEmbedder  # noqa: E402

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDP training for IResNet-100 + AdaFace")

    p.add_argument("--data-root", required=True, help="dir with train.rec/train.idx/property")
    p.add_argument("--output-dir", required=True, help="dir for checkpoints + logs")

    p.add_argument("--num-classes", type=int, default=0, help="0 = infer from property file")
    p.add_argument("--embedding-size", type=int, default=512)

    p.add_argument("--batch-size", type=int, default=128, help="per-GPU batch size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1, help="base LR for total batch ~512")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--poly-power", type=float, default=2.0)

    # AdaFace hyperparameters (paper defaults)
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--h", type=float, default=0.333)
    p.add_argument("--scale", type=float, default=64.0)
    p.add_argument("--t-alpha", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=-1, help="-1 = auto from SLURM CPUs")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sync-bn", action="store_true", help="convert BN to SyncBatchNorm")

    p.add_argument("--log-interval", type=int, default=50, help="steps between log lines")
    p.add_argument("--save-interval", type=int, default=2000, help="steps between mid-epoch saves; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup / logging
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    return rank, world_size, local_rank


def setup_logging(is_main: bool, output_dir: Path) -> None:
    level = logging.INFO if is_main else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if is_main:
        handlers.append(logging.FileHandler(output_dir / "train.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def poly_lr_with_warmup(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, power: float = 2.0
) -> float:
    """Linear warmup to ``base_lr`` over ``warmup_steps``, then polynomial decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return base_lr * ((1.0 - progress) ** power)


# ---------------------------------------------------------------------------
# Checkpoint I/O (atomic)
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(
    path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    map_location: str,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=map_location)
    model.module.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler.is_enabled() and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    setup_logging(is_main, output_dir)

    device = torch.device(f"cuda:{local_rank}")

    # Resolve worker count: divide the SLURM CPU allocation across the local procs.
    if args.num_workers < 0:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        num_workers = max(2, cpus // max(1, local_world))
    else:
        num_workers = args.num_workers

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    use_amp = amp_dtype != torch.float32
    # GradScaler is only needed for fp16; bf16 has enough dynamic range without it.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and use_amp))

    if is_main:
        logger.info("world_size=%d  local_rank=%d  device=%s", world_size, local_rank, device)
        logger.info("per-GPU batch=%d  effective batch=%d  num_workers=%d",
                    args.batch_size, args.batch_size * world_size, num_workers)
        logger.info("amp=%s  grad_scaler=%s", args.amp_dtype, scaler.is_enabled())

    # ---- data (auto-detect RecordIO vs WebDataset) --------------------------
    import glob as _glob
    _g = _glob
    if _g.glob(os.path.join(args.data_root, "glint360k-*.tar.gz")):
        if is_main:
            logger.info("Detected WebDataset shards (WDS format)")
        dataset, sampler, loader = build_wds_dataloader(
            shard_dir=args.data_root,
            batch_size=args.batch_size,
            num_workers=num_workers,
            distributed=True,
            seed=args.seed,
        )
    else:
        if is_main:
            logger.info("Detected RecordIO format (train.rec / train.idx)")
        dataset, sampler, loader = build_dataloader(
            root_dir=args.data_root,
            batch_size=args.batch_size,
            num_workers=num_workers,
            distributed=True,
            seed=args.seed,
        )

    num_classes = args.num_classes or dataset.num_classes
    if not num_classes:
        raise ValueError(
            "num_classes could not be inferred from the property file; "
            "pass --num-classes explicitly."
        )
    if is_main:
        logger.info("dataset: %d images, %d classes", len(dataset) if hasattr(dataset, '__len__') else 0, num_classes)

    # ---- model --------------------------------------------------------------
    torch.manual_seed(args.seed)  # identical init across ranks (DDP broadcasts anyway)
    model = FaceEmbedder(
        num_classes=num_classes,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        m=args.margin,
        h=args.h,
        s=args.scale,
        t_alpha=args.t_alpha,
    ).to(device)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,  # per-rank BN/norm stats (standard for face rec)
        find_unused_parameters=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    # ---- resume -------------------------------------------------------------
    start_epoch, global_step = 0, 0
    latest = output_dir / "latest.pth"
    if latest.is_file():
        start_epoch, global_step = load_checkpoint(
            latest, model, optimizer, scaler, map_location=f"cuda:{local_rank}"
        )
        if is_main:
            logger.info("resumed from %s at epoch=%d step=%d", latest, start_epoch, global_step)
    dist.barrier()

    # ---- train loop ---------------------------------------------------------
    model.train()
    window_start = time.time()
    window_loss = 0.0
    window_count = 0

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = poly_lr_with_warmup(global_step, total_steps, warmup_steps, args.lr, args.poly_power)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images, labels)
                loss = criterion(logits, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            global_step += 1
            window_loss += loss.item()
            window_count += 1

            if is_main and global_step % args.log_interval == 0:
                elapsed = time.time() - window_start
                imgs_per_sec = (window_count * args.batch_size * world_size) / max(elapsed, 1e-6)
                logger.info(
                    "epoch %d | step %d/%d | loss %.4f | lr %.5f | %.0f img/s",
                    epoch, global_step, total_steps, window_loss / max(window_count, 1), lr, imgs_per_sec,
                )
                window_start = time.time()
                window_loss = 0.0
                window_count = 0

            # Mid-epoch checkpoint (rank 0 only; no barrier so other ranks keep going).
            if is_main and args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(latest, model, optimizer, scaler, epoch, global_step, args)

        # End-of-epoch checkpoints.
        if is_main:
            save_checkpoint(latest, model, optimizer, scaler, epoch + 1, global_step, args)
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pth", model, optimizer, scaler,
                            epoch + 1, global_step, args)
            logger.info("saved checkpoints for epoch %d", epoch + 1)
        dist.barrier()

    # ---- final: backbone-only weights for export ----------------------------
    if is_main:
        backbone_path = output_dir / "backbone_final.pth"
        torch.save(model.module.backbone.state_dict(), backbone_path)
        logger.info("training complete; backbone saved to %s", backbone_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
