"""Model definitions for the facial-recognition embedding pipeline.

This module contains three things:

1. ``IResNet`` / ``iresnet100`` -- the "Improved ResNet" backbone used throughout
   the deep face-recognition literature (insightface / ArcFace / AdaFace). It maps
   an aligned ``112 x 112`` RGB face chip to a raw ``num_features``-dimensional
   feature vector (512 by default).
2. ``AdaFace`` -- the adaptive-margin classification head from
   *AdaFace: Quality Adaptive Margin for Face Recognition* (Kim et al., CVPR 2022,
   https://arxiv.org/abs/2204.00964). It uses the feature norm as an image-quality
   proxy and adapts the angular + additive margins accordingly.
3. ``FaceEmbedder`` -- a thin wrapper that ties the backbone and the head together,
   decomposing the backbone output into a direction (the embedding) and a magnitude
   (the norm, i.e. the quality signal) before handing both to the head.

For *deployment* we only ever use the backbone (``FaceEmbedder.backbone``): the
512-D vector it produces is L2-normalized at inference time and compared with cosine
similarity. The AdaFace head exists purely to shape the embedding space during
training and is discarded afterwards.

The numerics of the AdaFace head are deliberately forced to fp32 (autocast disabled
inside ``AdaFace.forward``) because it relies on ``acos``/``cos`` near +/-1, which is
unstable in fp16/bf16. The backbone itself runs happily under autocast.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "IResNet",
    "IBasicBlock",
    "iresnet100",
    "AdaFace",
    "FaceEmbedder",
    "l2_normalize",
]


# ---------------------------------------------------------------------------
# Convolution helpers (bias-free, matching the insightface IR-variant blocks)
# ---------------------------------------------------------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding, no bias (BN follows)."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, no bias (used in the downsample shortcut)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# Improved Basic Block (pre-activation, BN-Conv-BN-PReLU-Conv-BN)
# ---------------------------------------------------------------------------
class IBasicBlock(nn.Module):
    """The 'improved' residual block used by insightface backbones.

    Layout (pre-activation, no ReLU on the residual sum):

        identity = x
        out = bn1(x)
        out = conv1(out)            # 3x3, stride 1
        out = bn2(out)
        out = prelu(out)
        out = conv2(out)            # 3x3, stride = `stride`
        out = bn3(out)
        if downsample: identity = downsample(x)
        out += identity

    This differs from a torchvision BasicBlock: there is a BN *before* the first
    conv, a single PReLU activation, and no activation after the addition.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("IBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


# ---------------------------------------------------------------------------
# Improved ResNet backbone
# ---------------------------------------------------------------------------
class IResNet(nn.Module):
    """Improved ResNet backbone producing a flat feature embedding.

    Spatial progression for a 112x112 input:
        stem (stride 1) -> 112x112
        layer1 (stride 2) -> 56x56
        layer2 (stride 2) -> 28x28
        layer3 (stride 2) -> 14x14
        layer4 (stride 2) -> 7x7
    The final 7x7 = ``fc_scale`` feature map is flattened (512 * 49 = 25088) and
    projected to ``num_features`` by a single linear layer, then standardized by a
    frozen-scale BatchNorm1d.
    """

    fc_scale: int = 7 * 7

    def __init__(
        self,
        block: type[IBasicBlock],
        layers: list[int],
        num_features: int = 512,
        dropout: float = 0.0,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1

        # Stem: 3x3 stride-1 conv keeps the 112x112 resolution (insightface style,
        # as opposed to the 7x7 stride-2 stem of a vanilla ImageNet ResNet).
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        # Frozen-scale output BN: it standardizes the embedding (zero mean / unit
        # variance per dim) but its affine weight is pinned to 1 and frozen, so the
        # *magnitude* signal that AdaFace depends on is preserved while still
        # benefiting from running-statistics centering.
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None and m.weight.requires_grad:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if zero_init_residual:
            # Initializing the last BN in each residual branch to zero makes each
            # block start as an identity, which can help very deep nets converge.
            for m in self.modules():
                if isinstance(m, IBasicBlock) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0.0)

    def _make_layer(
        self,
        block: type[IBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(num_features: int = 512, dropout: float = 0.0, **kwargs) -> IResNet:
    """IResNet-100: blocks = [3, 13, 30, 3] (3 + 13 + 30 + 3 = 49 blocks * 2 convs
    + stem + fc ~= 100 weight layers)."""
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features, dropout=dropout, **kwargs)


# ---------------------------------------------------------------------------
# AdaFace head
# ---------------------------------------------------------------------------
def l2_normalize(x: Tensor, axis: int = 1, eps: float = 1e-12) -> Tensor:
    """L2-normalize ``x`` along ``axis`` with a numerically safe denominator."""
    norm = torch.norm(x, p=2, dim=axis, keepdim=True).clamp_min(eps)
    return x / norm


class AdaFace(nn.Module):
    """AdaFace adaptive-margin head (CVPR 2022).

    Given a *unit-norm* embedding direction and its (pre-normalization) feature
    norm, AdaFace turns the norm into a per-sample quality scalar and uses it to
    modulate both an angular margin and an additive (cosine) margin on the target
    class logit:

        quality  z_hat = clip( (||z|| - mu) / (sigma + eps) * h, -1, 1 )
        g_angle  = -m * z_hat          (added inside the arccos)
        g_add    =  m * z_hat + m      (subtracted from the cosine)

    where ``mu`` and ``sigma`` are EMA estimates of the batch feature-norm mean and
    std (so the quality scalar is self-calibrating over training). High-quality
    samples (large norm) get a larger effective margin -> hard samples are
    emphasized; low-quality samples get a smaller / negative margin -> likely-noisy
    hard samples are de-emphasized.

    The whole computation is forced to fp32 (autocast disabled) for ``acos``/``cos``
    stability.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 360232,
        m: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = eps

        # Class-center weight matrix (embedding_size, num_classes). Columns are the
        # per-identity prototypes; logits are cosine(embedding, prototype).
        self.kernel = nn.Parameter(torch.empty(embedding_size, num_classes))
        nn.init.normal_(self.kernel, mean=0.0, std=0.01)

        # EMA buffers for the feature-norm distribution. Initialized to the paper's
        # defaults (mean 20, std 100); they converge to the true batch statistics
        # within a few hundred steps via the t_alpha update.
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(self, embeddings: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        # Force fp32 regardless of any surrounding autocast context: acos/cos near
        # the +/-1 boundary are numerically fragile in reduced precision.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            embeddings = embeddings.float()
            kernel_norm = l2_normalize(self.kernel.float(), axis=0)

            # Cosine similarity to every class center, clamped away from +/-1 so the
            # subsequent acos has a valid gradient.
            cosine = torch.mm(embeddings, kernel_norm)
            cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

            # ---- image-quality indicator from the feature norm --------------------
            safe_norms = norms.float().clamp(0.001, 100).detach()

            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std(unbiased=False)
                self.batch_mean.mul_(1 - self.t_alpha).add_(self.t_alpha * mean)
                self.batch_std.mul_(1 - self.t_alpha).add_(self.t_alpha * std)

            margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
            margin_scaler = (margin_scaler * self.h).clamp(-1, 1)  # shape (B, 1)

            # One-hot over the target classes; (B, num_classes). Multiplying by the
            # (B, 1) scaler broadcasts so only the target column is shifted.
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

            # ---- adaptive angular margin: cos(theta + g_angle) --------------------
            g_angle = -self.m * margin_scaler
            theta = torch.acos(cosine)
            theta_m = (theta + one_hot * g_angle).clamp(self.eps, math.pi - self.eps)
            cosine = torch.cos(theta_m)

            # ---- adaptive additive margin: cosine - g_add -------------------------
            g_add = self.m * margin_scaler + self.m
            cosine = cosine - one_hot * g_add

            return cosine * self.s


# ---------------------------------------------------------------------------
# Full trainable model: backbone + head
# ---------------------------------------------------------------------------
class FaceEmbedder(nn.Module):
    """Backbone + AdaFace head.

    ``forward(images, labels)`` returns scaled margin logits suitable for a plain
    ``CrossEntropyLoss``. The backbone output is split into:
        norms      = ||feature||_2          (quality signal)
        embeddings = feature / norms         (unit direction)
    and both are passed to the head. For deployment, ignore the head entirely and
    export ``self.backbone`` (the raw 512-D feature is what downstream cosine
    matching consumes, normalized at inference time).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        dropout: float = 0.0,
        m: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.backbone = iresnet100(num_features=embedding_size, dropout=dropout)
        self.head = AdaFace(
            embedding_size=embedding_size,
            num_classes=num_classes,
            m=m,
            h=h,
            s=s,
            t_alpha=t_alpha,
        )

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        features = self.backbone(images)
        features = features.float()
        norms = torch.norm(features, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        embeddings = features / norms
        return self.head(embeddings, norms, labels)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = iresnet100(num_features=512, dropout=0.0)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)
    print(f"iresnet100 parameters : {n_params:,}")
    print(f"input shape           : {tuple(dummy.shape)}")
    print(f"embedding shape       : {tuple(out.shape)}")

    embedder = FaceEmbedder(num_classes=1000, embedding_size=512)
    logits = embedder(dummy, torch.tensor([3, 17]))
    print(f"FaceEmbedder logits   : {tuple(logits.shape)} (expected (2, 1000))")
