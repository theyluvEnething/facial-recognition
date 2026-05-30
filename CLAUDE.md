# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

State-of-the-art biometric facial recognition pipeline using ArcFace loss. Designed for distributed multi-GPU training on the **Leonardo Supercomputer** (CINECA) and optimized for local inference on **NVIDIA RTX 5070** hardware. Generates 512-dimensional vector embeddings consumed by FAISS/Milvus vector databases for 1:N identification.

## Tech Stack

- **Training**: PyTorch with DistributedDataParallel (DDP) over InfiniBand, SLURM job scheduler, Dixie configuration manager
- **Model backbone**: ResNet100 or Vision Transformer (ViT) with ArcFace (Additive Angular Margin Loss) head
- **Input**: 112×112 aligned face chips
- **Deployment**: ONNX export → NVIDIA TensorRT optimization (FP16/INT8 quantization)
- **Vector search**: FAISS / Milvus downstream
- **Environment split**: Training on Leonardo (HPC Linux + CUDA), inference on local Windows + RTX 5070

## Repository Structure (Planned)

```
src/
  dataset.py          # PyTorch DataLoader with augmentation pipeline
  model.py            # Backbone (ResNet100/ViT) + ArcFace head
  train_leonardo.py   # Multi-node DDP training orchestration
  export.py           # ONNX conversion + TensorRT optimization
  inference.py        # Local inference + benchmarking
scripts/
  job_submit.sh       # SLURM batch job submission
data/                 # Git-ignored; raw/ and processed/ subdirs
requirements.txt      # Python dependencies
```

## Development Notes

- **Dual-environment workflow**: Code is authored/tested locally on Windows, then deployed to Leonardo (HPC Linux) for training. Keep filesystem path differences and SLURM environment constraints in mind.
- **Local development** happens on Windows 10 with PowerShell as the shell and an RTX 5070 GPU.
- **requirements.txt** does not exist yet — it should pin PyTorch (CUDA builds), torchvision, onnx, onnxruntime, TensorRT Python bindings, and FAISS/Milvus client libraries.
- The `.gitignore` follows standard Python conventions (covers bytecode, virtualenvs, pytest cache, mypy, ruff, Jupyter).
- MIT licensed.
