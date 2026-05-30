# Dataset Acquisition & Indexing — Glint360K on Leonardo

This document is the **Task 1** blueprint: how to obtain a massive face-recognition
training set, what its on-disk binary format looks like, and how the `src/dataset.py`
reader streams it from Leonardo's `$SCRATCH` parallel filesystem **without choking the
metadata servers**.

---

## 1. Which dataset

| Dataset       | Identities | Images       | Approx. size | Notes |
|---------------|-----------:|-------------:|-------------:|-------|
| **Glint360K** | 360,232    | 17,091,657   | ~100+ GB     | **Recommended.** Cleanest large public set; SOTA baselines. |
| WebFace42M    | ~2,000,000 | ~42,000,000  | ~500+ GB     | Larger/heavier; only if you need maximum scale and have the time/disk. |

**Use Glint360K.** It is the sweet spot for a hackathon: 17M images over 360K
identities is more than enough to train a strong IResNet-100, it fits comfortably on
`$SCRATCH` (which has no quota), and the published baselines (ResNet-100 backbone,
CosFace `m=0.4`) reach IJB-C TAR@FAR=1e-4 around 97.3 and Megaface ~99.1, so you have
a clear correctness target.

`--num-classes` for Glint360K is **360232** (auto-detected from the `property` file by
the dataset code).

---

## 2. Where to get it

Glint360K is released by the **insightface** team via their `partial_fc` project:

- Repository: `https://github.com/deepinsight/insightface` →
  `recognition/partial_fc` (dataset zoo + unpack instructions).
- **Academic Torrents** (most reliable; the Baidu mirrors are slow/flaky):
  `https://academictorrents.com/details/e5f46ee502b9e76da8cc3a0e4f7c17e4000c7b1e`

It is distributed as a **multi-part archive** (`glint360k_00 … glint360k_06`).
Concatenate then untar — *the trailing `-` is required*:

```bash
cat glint360k_* | tar -xzvf -      # don't forget the final '-'
```

Per-part and post-extraction MD5s (verify these):

```
# archive parts
cf7433cbb915ac422230ba33176f4625  glint360k_00
589a5ea3ab59f283d2b5dd3242bc027a  glint360k_01
8d54fdd5b1e4cd55e1b9a714d76d1075  glint360k_02
cd7f008579dbed9c5af4d1275915d95e  glint360k_03
64666b324911b47334cc824f5f836d4c  glint360k_04
a318e4d32493dd5be6b94dd48f9943ac  glint360k_05
c3ae1dcbecea360d2ec2a43a7b6f1d94  glint360k_06

# extracted training files
5d9cd9f262ec87a5ca2eac5e703f7cdf  train.idx
8483be5af6f9906e19f85dee49132f8e  train.rec
```

After extraction the directory looks like:

```
glint360k/
├── train.rec        # all images, packed (the big file)
├── train.idx        # record-id -> byte-offset table
├── property         # "360232,112,112"  (num_classes, H, W)
├── agedb_30.bin     # validation pair sets (optional, for eval)
├── calfw.bin
├── cfp_ff.bin
├── cfp_fp.bin
├── cplfw.bin
├── lfw.bin
└── vgg2_fp.bin
```

Training uses **only `train.rec` + `train.idx` + `property`**. The `*.bin` files are
held-out verification pair sets you can use later to measure accuracy.

> **License / provenance.** Glint360K (and models trained on it) are released for
> **non-commercial research purposes only**. Confirm this fits your intended use
> before training or distributing weights, and keep the attribution to insightface.

---

## 3. Downloading *on Leonardo*

Compute nodes have **no internet**, and login nodes kill any process exceeding
**10 minutes of CPU time** — too short for a 100 GB download/extract. The on-cluster
HTTP proxy is for low-bandwidth API traffic only; do **not** pull large data through
it (it restarts on the same 10-minute CPU clock and drops TCP connections).

Instead, grab an interactive serial session, which has internet **and** no 10-minute
limit:

```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
```

Then, inside that session, download and extract straight into `$SCRATCH`:

```bash
mkdir -p "$SCRATCH/datasets/glint360k"
cd "$SCRATCH/datasets/glint360k"

# ... fetch the glint360k_* parts here (torrent client / wget from your mirror) ...

cat glint360k_* | tar -xzvf -          # trailing '-' !
md5sum -c <<'EOF'
5d9cd9f262ec87a5ca2eac5e703f7cdf  train.idx
8483be5af6f9906e19f85dee49132f8e  train.rec
EOF

rm -f glint360k_0*                     # reclaim space once extraction verifies
```

The training job (`scripts/job_submit.sh`) points `--data-root` at exactly this path,
`$SCRATCH/datasets/glint360k`.

---

## 4. The binary format (MXNet RecordIO)

insightface packs all 17M JPEGs into a single MXNet **RecordIO** stream. `src/dataset.py`
parses it directly — **no MXNet dependency** (MXNet is retired and awkward to install on
an offline node).

### 4.1 Record framing (`train.rec`)

Records are concatenated; each is framed as:

```
[ magic   : uint32 LE = 0xCED7230A ]
[ lrecord : uint32 LE              ]   # top 3 bits = cflag, low 29 bits = length
[ payload : <length> bytes         ]
[ padding : up to a 4-byte boundary ]
```

- `cflag = (lrecord >> 29) & 7` marks records split across chunks. insightface face
  packs never use this, so the reader treats `cflag != 0` as an error rather than
  silently truncating.
- `length = lrecord & ((1 << 29) - 1)` is the payload byte count.

### 4.2 The index (`train.idx`)

A plain text file, one line per record: `"<record_id>\t<byte_offset>"`. The reader
loads it into a `{record_id: offset}` dict once, giving **O(1) `seek()`** to any record
— no scanning, no per-image `stat()`.

### 4.3 Payload layout (`IRHeader` + image)

Each payload is an `IRHeader` followed by the encoded JPEG:

```
IRHeader = struct "IfQQ" (24 bytes): (flag, label, id, id2)
  flag == 0 : single label = `label`;           image = payload[24:]
  flag  > 0 : `flag` float32 labels follow header; image = payload[24 + flag*4 :]
```

### 4.4 The meta-header convention

insightface stores a **meta-header as record 0**: its `flag > 0` and its float label is
`[num_images + 1, ...]`. The real image records are therefore `range(1, num_images + 1)`.
`src/dataset.py` detects this (and falls back to "every key is an image" for packs that
don't follow it). Per-image records normally have `flag == 0`, so the label is the
single identity index in `[0, 360231]`.

### 4.5 Alternatives (for context)

- **WebDataset (`.tar` shards)**: some releases ship the same content as ~16 GB tar
  shards of `{jpg, cls}` pairs. This streams well too, but RecordIO + a single `.idx`
  gives cheaper random access for a `DistributedSampler`, so we standardize on it.
- **Raw `ImageFolder` of 17M JPEGs**: avoid on Lustre. Millions of tiny files turn
  every epoch's directory walks and `open()`s into a metadata-server storm — the exact
  failure mode the packed format prevents.

---

## 5. Streaming from `$SCRATCH` without hurting the filesystem

Leonardo's `$SCRATCH` is a fast Lustre filesystem. Lustre is excellent at large
sequential/strided reads from a few files and **terrible** under metadata pressure
(lots of `stat`/`open`/`readdir`). The pipeline is built around that reality:

1. **One packed file, not millions.** The entire training set is two files
   (`train.rec`, `train.idx`) plus `property`. Opening the dataset touches a handful of
   inodes total, regardless of the 17M images inside.
2. **O(1) offset lookups, zero stat storms.** Random access is a `dict` lookup +
   `seek()` into `train.rec`. The code never calls `os.path.exists` / `stat` per sample.
3. **One file descriptor per worker.** `worker_init_fn` opens a *fresh* handle to
   `train.rec` in each DataLoader worker (and the parent closes its handle after reading
   the meta-header), so forked workers never share a file offset — no seek contention,
   no locking. `persistent_workers=True` keeps those handles open across epochs.
4. **Stripe the big file across OSTs.** Before/after copying `train.rec` in, set Lustre
   striping so reads parallelize across storage targets:

   ```bash
   lfs setstripe -c 4 "$SCRATCH/datasets/glint360k/train.rec"
   # (set the stripe count on the file/dir before writing the data into it)
   ```

5. **Let the page cache work for you.** With ~480 GB of RAM on the 4-GPU allocation, a
   large fraction of the hot `.rec` pages stay cached after the first epoch, so later
   epochs read mostly from memory.
6. **Tune `num_workers` to the cores.** The job requests 32 CPUs; `train_leonardo.py`
   auto-splits them across the 4 GPU processes (`--num-workers -1` → ~8 workers/GPU),
   with `pin_memory=True` and `prefetch_factor=4` to keep the host→device copy and the
   GPUs saturated.
7. **Verify integrity once.** Check the MD5s above after extraction; a corrupt `.rec`
   surfaces as a decode error deep in training otherwise.

---

## 6. Verifying the pack

A quick sanity check using the reader in `src/dataset.py` (run on a login node or in an
interactive session, inside the pixi env):

```bash
pixi run python src/dataset.py "$SCRATCH/datasets/glint360k"
```

Expected output: a sample count of ~17,091,657, `num classes` of 360232, and one decoded
sample tensor of shape `(3, 112, 112)` with values in `[-1, 1]`. If the count or class
number is off, re-check that `train.idx`, `train.rec`, and `property` all came from the
same extraction.
