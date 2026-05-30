"""Lightweight, PyTorch-free face-embedding inference.

Loads an aligned 112x112 face image, preprocesses it identically to training
(BGR->RGB, scale to [-1, 1], CHW, batch dim), runs a forward pass through either an
ONNX model or a native TensorRT engine, and prints the raw 512-D embedding.

This deliberately depends only on ``numpy`` + ``opencv`` + ``onnxruntime`` (plus
optional ``tensorrt``/``pycuda``) so it can run on the local RTX 5070 box without a
full training stack.

    pixi run python src/inference.py --model model.onnx --image face.jpg
    pixi run python src/inference.py --model model.engine --image face.jpg --l2-normalize

ONNX Runtime is the primary path: its TensorRT execution provider gives FP16
TensorRT acceleration with zero manual buffer management, falling back to CUDA then
CPU. The native TensorRT path is provided for parity with the export script's engine
output.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

_IMG_SIZE = 112
_PIXEL_MEAN = 127.5
_PIXEL_STD = 127.5


# ---------------------------------------------------------------------------
# Preprocessing (must match dataset.py / training)
# ---------------------------------------------------------------------------
def preprocess(image_path: str) -> np.ndarray:
    """Return a contiguous ``(1, 3, 112, 112)`` float32 array in [-1, 1]."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != _IMG_SIZE or img.shape[1] != _IMG_SIZE:
        img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - _PIXEL_MEAN) / _PIXEL_STD
    img = np.transpose(img, (2, 0, 1))[None, ...]  # (1, 3, 112, 112)
    return np.ascontiguousarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX Runtime backend
# ---------------------------------------------------------------------------
def run_onnx(model_path: str, blob: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    out = sess.run([output_name], {input_name: blob})[0]
    return out[0]


# ---------------------------------------------------------------------------
# Native TensorRT backend (TensorRT 10 API; needs pycuda)
# ---------------------------------------------------------------------------
def run_tensorrt(engine_path: str, blob: np.ndarray) -> np.ndarray:
    import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    out_shape = tuple(engine.get_tensor_shape(out_name))

    blob = np.ascontiguousarray(blob, dtype=np.float32)
    out_host = np.empty(out_shape, dtype=np.float32)

    d_in = cuda.mem_alloc(blob.nbytes)
    d_out = cuda.mem_alloc(out_host.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, blob, stream)
    context.set_tensor_address(in_name, int(d_in))
    context.set_tensor_address(out_name, int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out_host, d_out, stream)
    stream.synchronize()

    return out_host[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run face-embedding inference")
    p.add_argument("--model", required=True, help="path to .onnx or .engine/.trt")
    p.add_argument("--image", required=True, help="path to an aligned 112x112 face image")
    p.add_argument("--l2-normalize", action="store_true", help="L2-normalize the output")
    p.add_argument("--save-npy", default=None, help="optional path to save the embedding as .npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    blob = preprocess(args.image)

    lower = args.model.lower()
    if lower.endswith((".engine", ".trt")):
        embedding = run_tensorrt(args.model, blob)
    else:
        embedding = run_onnx(args.model, blob)

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if args.l2_normalize:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    np.set_printoptions(precision=6, suppress=True, threshold=np.inf)
    print(f"embedding dim : {embedding.shape[0]}")
    print(embedding)

    if args.save_npy:
        np.save(args.save_npy, embedding)
        print(f"[ok] saved embedding to {args.save_npy}")


if __name__ == "__main__":
    main()
