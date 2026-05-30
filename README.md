# facial-recognition-model
State-of-the-art biometric facial recognition model pipeline trained on the Leonardo supercomputer utilizing ArcFace loss, optimized for high-throughput vector database embedding and local inference deployment.

# Face Biometrics Core (Training & Deployment)

A state-of-the-art biometric facial recognition pipeline designed for distributed multi-GPU training on the **Leonardo Supercomputer** (CINECA) and optimized for high-performance edge inference on local hardware (NVIDIA RTX 5070). 

This repository handles the core model training, weight optimization, and vector embedding generation. The consumer-facing application layer is maintained in a separate repository.

---

## 🚀 System Architecture Overview

1. **Training (Leonardo Supercomputer):** Multi-node distributed training using PyTorch DistributedDataParallel (DDP) over InfiniBand architecture, optimized via Dixie configuration manager.
2. **Loss Function:** Additive Angular Margin Loss (ArcFace) maximizing geodesic distance on a hypersphere to yield distinct identity clustering.
3. **Local Deployment (RTX 5070):** FP16/INT8 quantized model exported via ONNX and optimized using NVIDIA TensorRT for low-latency inference.
4. **Downstream Pipeline:** 512-dimensional vector embedding generation feeding directly into a similarity-search vector database (FAISS/Milvus) for 1:N identification tasks.

---

## 📁 Repository Structure

```text
├── data/                   # Local dataset directory (Git ignored)
│   ├── raw/                # Original data tracks (e.g., MS-Celeb-1M / WebFace600K)
│   └── processed/          # Aligned and cropped face chips (112x112)
├── src/                    # Core source code
│   ├── dataset.py          # Custom PyTorch DataLoader with augmentation pipeline
│   ├── model.py            # Backbone network architecture (ResNet100 / ViT) + ArcFace head
│   ├── train_leonardo.py   # Distributed multi-GPU training orchestration
│   ├── export.py           # ONNX conversion and TensorRT optimization scripts
│   └── inference.py        # Local PC runtime and benchmarking suite
├── scripts/                # High-Performance Computing (HPC) utilities
│   └── job_submit.sh       # SLURM cluster execution script
├── requirements.txt        # Python dependencies
└── README.md