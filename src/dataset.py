"""High-throughput face dataset for insightface-style MXNet RecordIO packs.

The large public face datasets (Glint360K, WebFace, MS1MV*) ship as a *single*
``train.rec`` binary plus a ``train.idx`` offset table and a ``property`` metadata
file. This is exactly the layout we want on a Lustre parallel filesystem: one big
packed file means a handful of metadata operations instead of the millions you'd
get from an ImageFolder of loose JPEGs (which would hammer Leonardo's metadata
servers).

The canonical reader for this format is ``mxnet.recordio``. MXNet is retired and a
pain to install on an offline compute node, so this module re-implements the record
format from scratch -- no mxnet dependency, just ``struct`` + ``cv2`` + ``numpy``.

RecordIO on disk
----------------
``train.idx`` is a text file, one record per line: ``<record_id>\\t<byte_offset>``.
``train.rec`` is a concatenation of records. Each record is framed as::

    [ magic   : uint32 little-endian = 0xCED7230A ]
    [ lrecord : uint32 little-endian              ]   # 3-bit cflag | 29-bit length
    [ payload : <length> bytes                    ]
    [ padding : to a 4-byte boundary              ]

For image records the payload is itself an ``IRHeader`` followed by the encoded
image::

    IRHeader = struct "IfQQ" (24 bytes): (flag, label, id, id2)
      - flag == 0 : label is the single float in `label`; image = payload[24:]
      - flag  > 0 : there are `flag` float32 labels right after the header;
                    image = payload[24 + flag*4 :]

insightface packs a *meta header* as record 0 with ``flag > 0`` whose float label is
``[num_images + 1, ...]``; the real image records are ``range(1, num_images + 1)``.
We detect and honor that convention, falling back to "every key is an image" for
packs that don't use it.
"""

from __future__ import annotations

import os
import random
import struct

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

# Each worker process decodes images single-threaded; let the DataLoader provide
# parallelism instead of having OpenCV spawn its own thread pool per worker.
cv2.setNumThreads(0)

__all__ = [
    "RecordIOReader",
    "MXFaceDataset",
    "build_dataloader",
    "worker_init_fn",
]

# ---------------------------------------------------------------------------
# Record format constants
# ---------------------------------------------------------------------------
_REC_MAGIC = 0xCED7230A
_IRHEADER_FMT = "IfQQ"  # flag (uint32), label (float32), id (uint64), id2 (uint64)
_IRHEADER_SIZE = struct.calcsize(_IRHEADER_FMT)  # == 24

_PIXEL_MEAN = 127.5
_PIXEL_STD = 127.5
_IMG_SIZE = 112

# Interpolations cycled through by the resize-based augmentations, matching the
# AdaFace augmentation pipeline's "random interpolation" idea.
_INTERP = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4,
]


# ---------------------------------------------------------------------------
# RecordIO reader (no mxnet)
# ---------------------------------------------------------------------------
class RecordIOReader:
    """Random-access reader for an insightface ``.rec`` / ``.idx`` pair.

    The ``.idx`` file is parsed once into an in-memory ``{record_id: offset}`` dict,
    giving O(1) seeks. The file handle is opened lazily and is *per-process*: it is
    deliberately dropped on pickling (``__getstate__``) so every DataLoader worker
    opens its own handle in ``worker_init_fn`` and there is no shared file-offset
    contention between forked workers.
    """

    def __init__(self, rec_path: str, idx_path: str) -> None:
        self.rec_path = rec_path
        self.idx_path = idx_path
        self.offsets: dict[int, int] = {}
        with open(idx_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key_str, offset_str = line.split("\t")
                self.offsets[int(key_str)] = int(offset_str)
        self.keys: list[int] = sorted(self.offsets.keys())
        self._fh = None  # opened lazily, per process

    # -- file-handle lifecycle ------------------------------------------------
    def open(self) -> None:
        if self._fh is None:
            self._fh = open(self.rec_path, "rb")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fh"] = None  # never pickle an open file handle
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self.keys)

    # -- record access --------------------------------------------------------
    def read_record(self, key: int) -> bytes:
        """Return the raw payload bytes (IRHeader + image) for ``key``."""
        if self._fh is None:
            self.open()
        offset = self.offsets[key]
        self._fh.seek(offset)
        header = self._fh.read(8)
        magic, lrecord = struct.unpack("<II", header)
        if magic != _REC_MAGIC:
            raise ValueError(
                f"Bad RecordIO magic 0x{magic:08X} at offset {offset} "
                f"(expected 0x{_REC_MAGIC:08X}); .idx and .rec may be mismatched."
            )
        cflag = (lrecord >> 29) & 7
        length = lrecord & ((1 << 29) - 1)
        if cflag != 0:
            # cflag != 0 marks a record split across multiple chunks. insightface
            # face packs never use this; refuse rather than silently truncate.
            raise ValueError(
                f"Chunked RecordIO record (cflag={cflag}) is not supported."
            )
        return self._fh.read(length)

    @staticmethod
    def unpack(payload: bytes) -> tuple[np.ndarray, bytes]:
        """Split a payload into ``(label_array, image_bytes)``."""
        flag, label, _id, _id2 = struct.unpack(_IRHEADER_FMT, payload[:_IRHEADER_SIZE])
        body = payload[_IRHEADER_SIZE:]
        if flag > 0:
            label_arr = np.frombuffer(body[: flag * 4], dtype=np.float32).copy()
            image_bytes = body[flag * 4 :]
        else:
            label_arr = np.array([label], dtype=np.float32)
            image_bytes = body
        return label_arr, image_bytes


# ---------------------------------------------------------------------------
# AdaFace-style augmentations (operate on uint8 HxWxC RGB arrays)
# ---------------------------------------------------------------------------
def low_res_augmentation(img: np.ndarray) -> np.ndarray:
    """Downscale then upscale to simulate low-resolution capture.

    This is the augmentation most directly responsible for creating the
    quality variation AdaFace's norm-as-quality mechanism learns from.
    """
    h, w = img.shape[:2]
    ratio = random.uniform(0.2, 1.0)
    small_h = max(1, int(ratio * h))
    small_w = max(1, int(ratio * w))
    small = cv2.resize(img, (small_w, small_h), interpolation=random.choice(_INTERP))
    return cv2.resize(small, (w, h), interpolation=random.choice(_INTERP))


def crop_augmentation(img: np.ndarray) -> np.ndarray:
    """Random crop of a sub-region, resized back to the original size."""
    h, w = img.shape[:2]
    ratio = random.uniform(0.5, 1.0)
    crop_h = max(1, int(ratio * h))
    crop_w = max(1, int(ratio * w))
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    cropped = img[top : top + crop_h, left : left + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=random.choice(_INTERP))


def photometric_augmentation(img: np.ndarray) -> np.ndarray:
    """Random brightness / contrast jitter."""
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta = random.uniform(-25, 25)  # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _read_num_classes(property_path: str) -> int | None:
    """Parse ``num_classes`` from an insightface ``property`` file.

    Format is ``<num_classes>,<image_h>,<image_w>`` (e.g. ``360232,112,112``).
    """
    if not os.path.isfile(property_path):
        return None
    with open(property_path, "r") as f:
        content = f.read().strip()
    if not content:
        return None
    parts = content.split(",")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


class MXFaceDataset(Dataset):
    """Reads aligned 112x112 face chips from a RecordIO pack and returns
    ``(image_tensor, label)`` with image normalized to ``[-1, 1]`` in CHW order.

    Augmentations (training only) are applied to the raw uint8 image before
    normalization, each gated by its own probability.
    """

    def __init__(
        self,
        root_dir: str,
        low_res_prob: float = 0.2,
        crop_prob: float = 0.2,
        photometric_prob: float = 0.2,
        hflip_prob: float = 0.5,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.training = training
        self.low_res_prob = low_res_prob
        self.crop_prob = crop_prob
        self.photometric_prob = photometric_prob
        self.hflip_prob = hflip_prob

        rec_path = os.path.join(root_dir, "train.rec")
        idx_path = os.path.join(root_dir, "train.idx")
        if not (os.path.isfile(rec_path) and os.path.isfile(idx_path)):
            raise FileNotFoundError(
                f"Expected train.rec and train.idx in {root_dir!r}. "
                "See documentation/dataset_acquisition.md for how to obtain them."
            )

        self.reader = RecordIOReader(rec_path, idx_path)

        # Detect the insightface meta-header on record 0. If present, the float
        # label encodes [num_images + 1, ...] and the real images are
        # range(1, num_images + 1). Otherwise treat every key as an image.
        first_payload = self.reader.read_record(self.reader.keys[0])
        header_label, _ = RecordIOReader.unpack(first_payload)
        if header_label.size >= 2 and float(header_label[0]) > 1:
            self.indices: list[int] = list(range(1, int(header_label[0])))
        else:
            self.indices = list(self.reader.keys)

        # Close the handle opened above so the parent process holds no open fd at
        # fork time; each worker opens a fresh one in worker_init_fn.
        self.reader.close()

        inferred = _read_num_classes(os.path.join(root_dir, "property"))
        self.num_classes: int | None = inferred

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.crop_prob:
            img = crop_augmentation(img)
        if random.random() < self.low_res_prob:
            img = low_res_augmentation(img)
        if random.random() < self.photometric_prob:
            img = photometric_augmentation(img)
        if random.random() < self.hflip_prob:
            img = img[:, ::-1]
        return img

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        key = self.indices[index]
        payload = self.reader.read_record(key)
        label_arr, image_bytes = RecordIOReader.unpack(payload)
        label = int(label_arr[0])

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, HxWx3
        if img is None:
            raise ValueError(f"Failed to decode image for record key {key}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != _IMG_SIZE or img.shape[1] != _IMG_SIZE:
            img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        if self.training:
            img = self._augment(img)

        img = np.ascontiguousarray(img, dtype=np.float32)
        img = (img - _PIXEL_MEAN) / _PIXEL_STD  # -> [-1, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return tensor, label


# ---------------------------------------------------------------------------
# Worker init: per-worker file handle + per-(rank, worker) RNG seeding
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own RecordIO file handle and a distinct
    RNG seed (so augmentations differ across workers *and* across DDP ranks)."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if isinstance(dataset, MXFaceDataset):
        dataset.reader.close()
        dataset.reader.open()

    rank = int(os.environ.get("RANK", "0"))
    # torch.initial_seed() is already unique per worker; fold in the rank so two
    # ranks running the same worker_id still diverge.
    seed = (torch.initial_seed() % (2**31) + worker_id + rank * 100003) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloader(
    root_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool = True,
    seed: int = 0,
    **aug_kwargs,
) -> tuple[MXFaceDataset, object, DataLoader]:
    """Construct ``(dataset, sampler, loader)`` for training.

    ``pin_memory=True`` and ``persistent_workers`` keep the host->device copy and
    the per-worker file handles efficient across epochs; ``prefetch_factor=4`` keeps
    the GPUs fed. With DDP, a ``DistributedSampler`` shards and reshuffles per epoch
    (call ``sampler.set_epoch(epoch)`` in the loop); otherwise a ``RandomSampler``.
    """
    dataset = MXFaceDataset(root_dir, training=True, **aug_kwargs)

    sampler: object
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
    else:
        sampler = RandomSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataset, sampler, loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python dataset.py /path/to/glint360k")
        raise SystemExit(1)

    root = sys.argv[1]
    ds = MXFaceDataset(root, training=True)
    print(f"dataset root   : {root}")
    print(f"num samples    : {len(ds):,}")
    print(f"num classes    : {ds.num_classes}")
    img, lbl = ds[0]
    print(f"sample[0] image: {tuple(img.shape)} dtype={img.dtype} "
          f"min={img.min():.3f} max={img.max():.3f}")
    print(f"sample[0] label: {lbl}")
