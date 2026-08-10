#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=small-g
#SBATCH --job-name=bridge_64
# create the dir once, before the first submit:
#   mkdir -p /flash/project_465002822/sb-match/outputs/log
#SBATCH --output=/flash/project_465002822/sb-match/outputs/log/bridge_64_%j.out
#SBATCH --error=/flash/project_465002822/sb-match/outputs/log/bridge_64_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7            # Use --gpus-per-node*7 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G
# estimate ~14.5 h (5.6 h train + 8.9 h cache), derived from the 512px smoke run's
# 1.144 s/train step at bs=4 and 0.90 s/sim eval at bs=10, divided by the ~22x conv-FLOP
# ratio between the two build_models() branches. That extrapolation carries a 2-3x band,
# so the real range is 7-40 h -- hence the partition max rather than 24:00:00.
# See conf/experiment/bridge_64.yaml for the full derivation.
#SBATCH --time=48:00:00

# NOTE: train() has no resume (DESIGN CHOICE 9). If this job is killed it restarts from
# scratch. Per-drift loss on a kill is bounded by outer 0's pretrain, 6250 * 64 images
# = ~1.4 h here, which is why 64px tolerates the coarse checkpointing that 512px does not
# (31.8 h there).

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

FLASH_BASE=/flash/project_465002822/sb-match/
mkdir -p "${FLASH_BASE}/tmp"
export TMPDIR="${FLASH_BASE}/tmp"

# wandb runs online; the smoke run established that compute nodes reach api.wandb.ai.
# WANDB_DIR would otherwise default to the submission cwd (hydra runs with chdir=false)
# and WANDB_CACHE_DIR to ~/.cache/wandb, which eats the home quota.
# singularity forwards the host env, so these reach the container without SINGULARITYENV_.
export WANDB_DIR="${FLASH_BASE}/outputs"
export WANDB_CACHE_DIR="${FLASH_BASE}/wandb-cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

PROJECT_DIR=/project/project_465002822/sb-match/

SIF=/flash/project_465002822/containers/sb-match-20260810.sif

srun singularity run \
  -B /scratch/project_465002822 \
  -B /project/project_465002822 \
  -B /flash/project_465002822 \
  "${SIF}" \
  bash -c "PYTHONPATH=${PROJECT_DIR} python3 -u ${PROJECT_DIR}/main.py experiment=bridge_64"
