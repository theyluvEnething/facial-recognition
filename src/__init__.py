"""Face biometrics embedding pipeline — IResNet-100 backbone + AdaFace head.

Modules:
    dataset         — MXNet RecordIO reader + augmentations + DataLoader
    model           — IResNet100 backbone, AdaFace loss head, FaceEmbedder wrapper
    train_leonardo  — DistributedDataParallel training on Leonardo (4x A100)
    export          — ONNX export + TensorRT engine build
    inference       — Lightweight inference (ONNX Runtime / TensorRT)
"""
