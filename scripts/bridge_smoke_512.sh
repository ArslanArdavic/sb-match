#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=small-g
#SBATCH --job-name=bridge_smoke_512
# create the dir once, before the first submit:
#   mkdir -p /flash/project_465002822/sb-match/outputs/log
#SBATCH --output=/flash/project_465002822/sb-match/outputs/log/bridge_smoke_512_%j.out
#SBATCH --error=/flash/project_465002822/sb-match/outputs/log/bridge_smoke_512_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7            # Use --gpus-per-node*7 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00              # smoke run is ~10 min, the rest is headroom

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

FLASH_BASE=/flash/project_465002822/sb-match/
mkdir -p "${FLASH_BASE}/tmp"
export TMPDIR="${FLASH_BASE}/tmp"

# wandb runs online here; whether compute nodes reach api.wandb.ai is one of the things
# this smoke run is here to find out. wandb writes a local run dir either way.
# WANDB_DIR would otherwise default to the submission cwd (hydra runs with chdir=false)
# and WANDB_CACHE_DIR to ~/.cache/wandb, which eats the home quota.
# singularity forwards the host env, so these reach the container without SINGULARITYENV_.
export WANDB_DIR="${FLASH_BASE}/outputs"
export WANDB_CACHE_DIR="${FLASH_BASE}/wandb-cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

PROJECT_DIR=/project/project_465002822/sb-match/

SIF=/project/project_465002822/containers/sb-match-20260627.sif

srun singularity run \
  -B /scratch/project_465002822 \
  -B /project/project_465002822 \
  -B /flash/project_465002822 \
  "${SIF}" \
  bash -c "PYTHONPATH=${PROJECT_DIR} python3 -u ${PROJECT_DIR}/main.py experiment=smoke"