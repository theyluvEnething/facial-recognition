# Leonardo Supercomputer — Full Tutorial & Reference

**Presented by:** Simeon Harrison & Martin Pfister  
**Event:** Zero One Hack_01 — AI Factory Austria  
**Date:** 29–31 May 2026  
**Host:** Lumos Consulting @ AI Factory Austria, Vienna

---

## Table of Contents

1. [What is a Supercomputer?](#1-what-is-a-supercomputer)
2. [Top 500 & EuroHPC Systems](#2-top-500--eurohpc-systems)
3. [Typical Setup of a Supercomputer](#3-typical-setup-of-a-supercomputer)
4. [SSH: Secure Shell](#4-ssh-secure-shell)
5. [Our Leonardo Account](#5-our-leonardo-account)
6. [File Systems on Leonardo](#6-file-systems-on-leonardo)
7. [Pixi Package Manager](#7-pixi-package-manager)
8. [Singularity / Apptainer Containers](#8-singularity--apptainer-containers)
9. [SLURM: Job Scheduler](#9-slurm-job-scheduler)
10. [Job Script Examples](#10-job-script-examples)
11. [Useful SLURM Commands](#11-useful-slurm-commands)
12. [Internet Access on Compute Nodes](#12-internet-access-on-compute-nodes)
13. [Login Node CPU Time Limit](#13-login-node-cpu-time-limit)
14. [Quick-Start Workflow](#14-quick-start-workflow)
15. [Additional Resources](#15-additional-resources)
16. [Quick Reference Card](#16-quick-reference-card)

---

## 1. What is a Supercomputer?

A supercomputer is a high-performance computing (HPC) system that aggregates enormous computational power to solve problems that are too large for standard computers.

**Key concepts:**
- **Login nodes** — entry points where you prepare your work (compile code, download data, submit jobs)
- **Compute nodes** — the actual workers that run your computations (have GPUs, lots of RAM)
- **Storage** — shared file systems accessible from both login and compute nodes
- **Job scheduler** — allocates compute resources to users (SLURM on Leonardo)
- **Head node** — orchestrates everything

You **never** run heavy computations on the login nodes — you submit *jobs* that run on compute nodes.

---

## 2. Top 500 & EuroHPC Systems

Leonardo is a **top 10** supercomputer in the world.

### Top 500 (as of June 2025)

| Rank | System | Location | GPU Type |
|------|--------|----------|----------|
| #4 | **JUPITER** | Jülich, Germany | NVIDIA GH200 |
| #9 | **LUMI** | Kajaani, Finland | AMD MI250X |
| #10 | **LEONARDO** | Bologna, Italy | **NVIDIA A100** |

🔗 [https://www.top500.org/](https://www.top500.org/)

### AI Factory Austria (AI:AT)

- Part of the **EuroHPC Joint Undertaking (JU)**
- Funded by Horizon Europe & Austria (BMIMI / FFG)
- Provides access to EuroHPC systems for AI innovation
- See: [https://ai-at.eu/hpc-onboarding/](https://ai-at.eu/hpc-onboarding/)

---

## 3. Typical Setup of a Supercomputer

```
┌─────────────────────────────────────────────────┐
│                 HEAD NODE                        │
│        (login, job submission, orchestration)    │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ COMPUTE  │  │ COMPUTE  │  │ COMPUTE  │   ...  │
│  │ NODE 1   │  │ NODE 2   │  │ NODE 3   │        │
│  │ GPU: A100 │  │ GPU: A100 │  │ GPU: A100 │        │
│  └──────────┘  └──────────┘  └──────────┘        │
│                                                   │
├─────────────────────────────────────────────────┤
│              SHARED STORAGE ($HOME, $SCRATCH)     │
└─────────────────────────────────────────────────┘
```

- **Login nodes**: SSH into these to work
- **Compute nodes**: Run your jobs here (SLURM scheduler)
- **Storage**: Shared — any node can access your files

---

## 4. SSH: Secure Shell

### What is it?

SSH (Secure Shell) is a protocol for:
- **Remote access** to supercomputers
- **End-to-end encrypted** connections
- Running commands / getting a shell
- **File transfer** (SCP/SFTP)
- **Port forwarding / tunnels**

### SSH on Leonardo

Use any of the following login nodes:

```bash
ssh a08trc28@login01-ext.leonardo.cineca.it
ssh a08trc28@login02-ext.leonardo.cineca.it
ssh a08trc28@login05-ext.leonardo.cineca.it
ssh a08trc28@login07-ext.leonardo.cineca.it
```

**For this hackathon: 2FA is NOT used.** You connect directly with your password.

### Passwordless Login (SSH Key Setup)

To avoid typing your password every time, set up an SSH key:

```bash
# On your local machine — generate a key pair (if you don't already have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy the public key to Leonardo (enter password one last time)
ssh-copy-id a08trc28@login01-ext.leonardo.cineca.it

# Or manually:
cat ~/.ssh/id_ed25519.pub | ssh a08trc28@login01-ext.leonardo.cineca.it "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Now you can connect without a password prompt.

### SCP / SFTP — File Transfer

```bash
# Upload a file to Leonardo
scp my_script.py a08trc28@login01-ext.leonardo.cineca.it:/scratch/a08trc28/

# Download a file from Leonardo
scp a08trc28@login01-ext.leonardo.cineca.it:/scratch/a08trc28/results.txt .

# Upload a whole directory
scp -r my_folder/ a08trc28@login01-ext.leonardo.cineca.it:/scratch/a08trc28/

# Interactive file transfer (like FTP)
sftp a08trc28@login01-ext.leonardo.cineca.it
```

---

## 5. Our Leonardo Account

| Field | Value |
|-------|-------|
| **Username** | `a08trc28` |
| **Password** | `wyadPacVioj7` |
| **SSH host** | `login01-ext.leonardo.cineca.it` |
| **Partition** | `boost_usr_prod` |
| **Reservation** | `s_tra_ncc` |
| **Max GPUs per node** | 4 |
| **Max runtime per job** | 24 hours |
| **Max nodes per team** | 1 (reservation only enough for 1 node per team) |

---

## 6. File Systems on Leonardo

All file systems are **shared** across login and compute nodes — files saved in one place are visible everywhere.

| Variable | Path | Size | Persistence |
|----------|------|------|-------------|
| `$HOME` | `/users/a08trc28` | 50 GB | **Permanent**, backed up |
| `$SCRATCH` | `/scratch/a08trc28` | No quota | **Auto-deleted after 40 days** |
| `$PUBLIC` | `/public/a08trc28` | 50 GB | Share files between users |

### ⚠️ Important Rules

- **`$WORK` and `$FAST` are NOT available** during the hackathon — do not use them.
- **Use `$SCRATCH` for everything**: project files, data, models, training outputs.
- **Use `$HOME` only for**: config files, SSH keys, and the pixi binary.
- **`$CINECA_SCRATCH` auto-purges after 40 days** — don't store anything permanent there.

### Check Your Disk Usage

```bash
# Check home quota
quota -s

# Check scratch usage
du -sh /scratch/a08trc28/*
```

---

## 7. Pixi Package Manager

[pixi](https://pixi.sh/) is the **recommended package manager** for this hackathon. It's a fast, Rust-based tool from prefix.dev that:
- Combines **conda packages** (from conda-forge) with **PyPI support**
- Produces **deterministic lock files** (`pixi.lock`)
- Handles **CUDA/GPU dependencies** natively
- Is much faster than conda/mamba

### Installation on Leonardo

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Restart shell or source your profile
source ~/.bashrc
```

### Setup Cache to Avoid Filling $HOME

```bash
export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache
```

Add this to your `~/.bashrc` so it persists across logins:

```bash
echo 'export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache' >> ~/.bashrc
```

### Quick Start

```bash
# Initialize a new project
pixi init my-project
cd my-project

# Install Python from conda-forge
pixi add python

# Install PyPI packages
pixi add --pypi openai transformers datasets

# Install everything
pixi install

# Run a command in the pixi environment
pixi run python -c "print('Hello World!')"
```

### Multi-Environment Setup (CPU dev on laptop, GPU on Leonardo)

```toml
# pixi.toml
[project]
name = "zero-one-hack"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.12.*"
pytorch = ">=2.5"

[feature.dev.dependencies]
pytest = "*"
ruff = "*"
ipython = "*"

[feature.gpu.system-requirements]
cuda = "12"

[feature.gpu.dependencies]
pytorch-cuda = { version = "==12.*", channel = "pytorch" }

[environments]
default = ["dev"]
gpu = ["dev", "gpu"]
```

Use `pixi shell -e gpu` for training, `pixi shell -e default` for local development.

### Running in SLURM Jobs

```bash
# Option A: Run directly
pixi run python train.py

# Option B: Use --as-is to pass shell constructs
pixi run --as-is python train.py --batch-size 64 --epochs 100

# Option C: Specify a manifest path
pixi run --manifest-path /path/to/project python train.py
```

---

## 8. Singularity / Apptainer Containers

Most HPC systems **don't run Docker** — instead they use **Singularity** or **Apptainer** (highly related, nearly identical).

🔗 [https://docs.sylabs.io/guides/latest/user-guide/](https://docs.sylabs.io/guides/latest/user-guide/)

### Why not Docker?

- Docker requires root privileges (security risk on shared systems)
- Singularity runs **rootless** — it's designed for HPC
- Singularity can **convert Docker images** directly

### Pulling a Docker Image on Leonardo

Download on a login node or via an interactive session:

```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty \
  singularity pull vllm-openai-v0.21.0-cu129.sif \
  docker://docker.io/vllm/vllm-openai:0.21.0-cu129
```

### Running a Container

```bash
# Basic execution
singularity exec --nv container.sif python3 script.py

# Bind mount scratch directory
singularity exec --nv --bind $SCRATCH:/scratch container.sif python3 script.py

# Get a shell inside the container (for debugging)
singularity shell --nv container.sif
```

**Flags explained:**
- `--nv`: Enable NVIDIA GPU support (mounts CUDA drivers)
- `--bind`: Mount host directories into the container

---

## 9. SLURM: Job Scheduler

SLURM (Simple Linux Utility for Resource Management) is the **job scheduler** on Leonardo. It:
- Allocates compute nodes to users
- Manages job queues
- Enforces resource limits (GPUs, memory, time)

### Lifecycle of a Job

```
Prepare files    →    Submit job    →    Wait in queue    →    Run    →    Get results
(login node)          (sbatch)            (pending)            (running)    (output files)
```

### Anatomy of a SLURM Job Script

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod    # Which partition to use
#SBATCH --reservation=s_tra_ncc       # Hackathon reservation
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node
#SBATCH --gpus-per-task=1             # GPUs per task (up to 4)
#SBATCH --mem=120GB                   # RAM (120 GB × #GPUs)
#SBATCH --cpus-per-task=8             # CPUs (8 × #GPUs)
#SBATCH --time=0:30:00                # Time limit HH:MM:SS (max 24:00:00)

# Your commands here
python3 script.py
```

### Fair-Share Resource Rules

| GPUs | Memory | CPUs | Nodes |
|------|--------|------|-------|
| 1 | 120 GB | 8 | 1 |
| 2 | 240 GB | 16 | 1 |
| 4 | 480 GB | 32 | 1 |
| 4 per node × 2 nodes | 480 GB per node | 32 per node | 2 |

**Limitation:** The reservation only allows **1 node per team**. Multi-node jobs are possible outside the reservation but may wait longer.

### How SLURM Interprets Resource Flags

The relationship between `--nodes`, `--ntasks-per-node`, and `--gpus-per-task`:

```
Total GPUs = nodes × ntasks-per-node × gpus-per-task
Total CPUs = nodes × ntasks-per-node × cpus-per-task
Total RAM  = nodes × ntasks-per-node × mem
```

**Example:** `--nodes=2 --ntasks-per-node=1 --gpus-per-task=4`
→ Total: 2 nodes, 2 tasks, 8 GPUs

---

## 10. Job Script Examples

### 1-GPU Job — Using pixi

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

# Set cache directory
export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache

# Run command inside pixi environment
export RUN_COMMAND="/path/to/pixi run --as-is"
$RUN_COMMAND python3 script.py
```

> **Note:** Replace `/path/to/pixi` with the actual path. Find it with: `which pixi`

### 1-GPU Job — Using Singularity

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

# Run inside Singularity container
export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 2-GPU Job — Using Singularity

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 4-GPU Job — Using Singularity

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 2-Node Multi-GPU Job (outside reservation)

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
srun $CONTAINER python3 script.py
```

> **Note:** The reservation only guarantees 1 node per team. Multi-node jobs may wait longer.

### Interactive Session (for debugging)

```bash
srun --partition=boost_usr_prod --reservation=s_tra_ncc \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
  --mem=120GB --time=02:00:00 --pty /bin/bash
```

This gives you a **shell on a compute node** for 2 hours — useful for debugging GPU code interactively.

### Download Large Files (Interactive Session)

```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
```

This gives you a shell with internet access and a large temp directory for downloading models.

---

## 11. Useful SLURM Commands

```bash
# Submit a job
sbatch job.sh

# Check your submitted jobs
squeue --me

# Check all jobs on the partition
squeue --partition=boost_usr_prod

# Get detailed info on a specific job
scontrol show job <JOBID>

# Look at the output from a job
cat slurm-<JOBID>.out

# Follow output as the job runs (like tail -f)
tail -c +0 -f slurm-<JOBID>.out

# Cancel a job
scancel <JOBID>

# Cancel ALL your jobs
scancel -u a08trc28

# Check partition/node status
sinfo

# Get a shell on a node while a job is running
srun --overlap --pty --jobid=<JOBID> bash

# Hold a job (prevent it from starting)
scontrol hold <JOBID>

# Release a held job
scontrol release <JOBID>
```

### Understanding Job States

| State | Meaning |
|-------|---------|
| `PENDING` (PD) | Job is queued, waiting for resources |
| `RUNNING` (R) | Job is actively executing |
| `COMPLETED` (CD) | Job finished successfully |
| `FAILED` (F) | Job ended with an error |
| `CANCELLED` (CA) | Job was cancelled by user or admin |
| `TIMEOUT` (TO) | Job reached its time limit |

---

## 12. Internet Access on Compute Nodes

### ⚠️ Critical Constraint

**Compute nodes have NO DIRECT internet access.**

### Strategy

1. **Download large files on login nodes** (or via `srun --partition=lrd_all_serial`)
2. **Use the HTTP proxy** only for low-bandwidth traffic (API calls, small downloads)

### Proxy Setup

Add to your job script for low-bandwidth HTTP/HTTPS access:

```bash
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export http_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export https_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
```

### ⚠️ Proxy Limitations

- The proxy **restarts periodically** (10 min CPU time limit)
- **TCP connections will drop** when it restarts
- Only use for **low-bandwidth** traffic
- **Never download large models or datasets** through the proxy
- For reliable long-running API calls, handle reconnection in your code

### Complete Example with Proxy

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

# Set up proxy for low-bandwidth traffic
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export http_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export https_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425

# Run your command
python3 script.py
```

---

## 13. Login Node CPU Time Limit

Processes on **login nodes** have a **10 minute CPU time limit**.

If your process runs for more than 10 minutes of CPU time on a login node, it will be killed.

### Workaround: Interactive Session for Long Tasks

For longer interactive work (e.g., downloading large models, compiling), request an interactive session:

```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
```

This gives you a shell on a compute node (or serial node) with:
- Up to 4 hours of runtime
- 100 GB of temp storage (`$TMPDIR`)
- Internet access (for downloading)
- No 10-minute CPU limit

---

## 14. Quick-Start Workflow

### First Login

```bash
# Step 1: SSH into Leonardo
ssh a08trc28@login01-ext.leonardo.cineca.it
# (enter password: wyadPacVioj7)

# Step 2: (Optional) Set up SSH key for passwordless login (from your local machine)
# On local machine: ssh-copy-id a08trc28@login01-ext.leonardo.cineca.it

# Step 3: Install pixi
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# Step 4: Set pixi cache to scratch
echo 'export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache' >> ~/.bashrc
source ~/.bashrc
```

### Setting Up a Project

```bash
# Step 5: Navigate to scratch
cd /scratch/a08trc28

# Step 6: Initialize a pixi project
pixi init my-hackathon-project
cd my-hackathon-project

# Step 7: Add dependencies
pixi add python=3.12 pytorch torchvision torchaudio
pixi add --pypi transformers datasets scikit-learn jupyter

# Step 8: Install everything
pixi install
```

### Writing and Submitting a Job

```bash
# Step 9: Create your Python script
cat > train.py << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
EOF

# Step 10: Create your SLURM job script
cat > job.slurm << 'EOF'
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache

# Use pixi to run inside the environment
pixi run --manifest-path /scratch/a08trc28/my-hackathon-project python3 train.py
EOF

# Step 11: Submit the job
sbatch job.slurm

# Step 12: Monitor the job
squeue --me

# Step 13: Check output
cat slurm-*.out
```

### End-to-End LLM Inference Workflow

```bash
# Step 1: On LOGIN node — download the model
cd /scratch/a08trc28
pixi run --pypi transformers huggingface-hub
python3 download_model.py  # e.g., Qwen/Qwen2.5-7B-Instruct

# Step 2: Create inference script and SLURM job
# (see work/leonardo/ for examples)

# Step 3: Submit to compute node
sbatch job_inference.slurm

# Step 4: Read results
cat results/inference_output_*.txt
```

---

## 15. Additional Resources

| Resource | Link |
|----------|------|
| AI:AT HPC Onboarding Kit | [https://ai-at.eu/hpc-onboarding/](https://ai-at.eu/hpc-onboarding/) |
| Top 500 List | [https://www.top500.org/](https://www.top500.org/) |
| Pixi Documentation | [https://pixi.sh/](https://pixi.sh/) |
| Singularity Docs | [https://docs.sylabs.io/guides/latest/user-guide/](https://docs.sylabs.io/guides/latest/user-guide/) |
| EuroHPC JU | [https://eurohpc-ju.europa.eu/](https://eurohpc-ju.europa.eu/) |
| AI Factory Austria | [https://ai-at.eu/](https://ai-at.eu/) |

---

## 16. Quick Reference Card

### SSH Login
```bash
ssh a08trc28@login01-ext.leonardo.cineca.it
```
Password: `wyadPacVioj7`

### File Systems
| Path | Use For |
|------|---------|
| `/scratch/a08trc28` | **Everything** (data, code, models, outputs) |
| `/users/a08trc28` | Config, SSH keys, pixi binary only |

### SLURM Basics
| Command | What it does |
|---------|-------------|
| `sbatch job.sh` | Submit job |
| `squeue --me` | Check your jobs |
| `scancel <JOBID>` | Cancel job |
| `tail -f slurm-<JOBID>.out` | Watch live output |

### Resource Template
```
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1     # 1, 2, or 4
#SBATCH --mem=120GB            # 120 per GPU
#SBATCH --cpus-per-task=8      # 8 per GPU
#SBATCH --time=0:30:00         # HH:MM:SS
```

### Key Rules
1. ✅ Use `$SCRATCH` for everything
2. ❌ No `$WORK` or `$FAST`
3. ✅ Download large files on login nodes
4. ❌ No internet on compute nodes (use proxy for API calls only)
5. ✅ 10 min CPU limit on login nodes — use `srun --partition=lrd_all_serial` for longer tasks
6. ✅ Pixi for package management
7. ❌ Singularity instead of Docker on compute nodes

---

*Funded by the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 101253078. The JU receives support from the Horizon Europe Programme of the European Union and Austria (BMIMI / FFG).*
