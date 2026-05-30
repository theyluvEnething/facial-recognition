"""Export a trained IResNet-100 backbone to ONNX (and optionally TensorRT).

The training script saves several artifacts in the output dir:
  * ``latest.pth`` / ``epoch_NNN.pth`` -- full checkpoints (FaceEmbedder state_dict
    under the key ``"model"``, with ``backbone.*`` and ``head.*`` prefixes).
  * ``backbone_final.pth`` -- a plain backbone state_dict (no prefix).
This loader accepts either form.

We export *only the backbone* with a static ``1 x 3 x 112 x 112`` input. Static
shapes are what NVIDIA TensorRT wants for the best optimization, and the downstream
RTX 5070 inference path consumes the raw 512-D output (L2-normalized at inference).

    pixi run python src/export.py \\
        --checkpoint $SCRATCH/.../backbone_final.pth \\
        --onnx-out model.onnx

IMPORTANT: a TensorRT *engine* is hardware- and driver-specific. Build it on the
*deployment* GPU (the RTX 5070), NOT on Leonardo's A100. ``--build-trt`` is therefore
off by default; this script always emits portable ONNX and only builds an engine
when explicitly asked (and when the ``tensorrt`` Python package is importable).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import iresnet100  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_backbone(checkpoint_path: str, embedding_size: int = 512, device: str = "cpu") -> torch.nn.Module:
    """Build an iresnet100 and load weights from either a full FaceEmbedder
    checkpoint or a plain backbone state_dict."""
    model = iresnet100(num_features=embedding_size, dropout=0.0)
    ckpt = torch.load(checkpoint_path, map_location=device)

    raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    backbone_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        name = key
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("backbone."):
            # Full FaceEmbedder checkpoint: keep backbone.* only, drop head.*
            backbone_state[name[len("backbone."):]] = value
        elif name.startswith("head."):
            continue
        else:
            # Plain backbone state_dict.
            backbone_state[name] = value

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX export + verification
# ---------------------------------------------------------------------------
def export_onnx(model: torch.nn.Module, onnx_path: str, opset: int = 17) -> None:
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # static 1x3x112x112
    )
    print(f"[ok] wrote ONNX: {onnx_path}")


def verify_onnx(model: torch.nn.Module, onnx_path: str, tol: float = 1e-3) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not available; skipping verification")
        return

    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["embedding"], {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    status = "ok" if max_diff < tol else "FAIL"
    print(f"[{status}] max |torch - onnx| = {max_diff:.3e} (tol {tol:.0e})")


# ---------------------------------------------------------------------------
# Optional TensorRT engine build (run on the deployment GPU only)
# ---------------------------------------------------------------------------
def build_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True, workspace_gb: int = 4) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("[warn] tensorrt python package not found. Build the engine on the "
              "deployment GPU, e.g.:")
        print(f"    trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[trt] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[ok] wrote TensorRT engine: {engine_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export backbone to ONNX / TensorRT")
    p.add_argument("--checkpoint", required=True, help="path to .pth (full ckpt or backbone)")
    p.add_argument("--onnx-out", required=True, help="output .onnx path")
    p.add_argument("--embedding-size", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true", help="skip onnxruntime verification")
    p.add_argument("--build-trt", action="store_true",
                   help="also build a TensorRT engine (run on the DEPLOYMENT GPU only)")
    p.add_argument("--engine-out", default=None, help="output .engine path (with --build-trt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_backbone(args.checkpoint, embedding_size=args.embedding_size, device="cpu")
    export_onnx(model, args.onnx_out, opset=args.opset)
    if not args.no_verify:
        verify_onnx(model, args.onnx_out)
    if args.build_trt:
        engine_out = args.engine_out or os.path.splitext(args.onnx_out)[0] + ".engine"
        build_tensorrt(args.onnx_out, engine_out, fp16=True)


if __name__ == "__main__":
    main()
