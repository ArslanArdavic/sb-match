#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=small-g
#SBATCH --job-name=forward_only
#SBATCH --output=tests/outputs/log/forward_only_train_full_size_%j.out
#SBATCH --error=tests/outputs/log/forward_only_train_full_size_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7            # Use --gpus-per-node*7 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G
#SBATCH --time=72:00:00              # time limit

mkdir -p tests/outputs/log

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings


FLASH_BASE=/flash/project_465002822/sb-match/
mkdir -p "${FLASH_BASE}/tmp"
export TMPDIR="${FLASH_BASE}/tmp"

PROJECT_DIR=/project/project_465002822/sb-match/

SIF=/project/project_465002822/containers/sb-match-20260627.sif

srun singularity run \
  -B /scratch/project_465002822 \
  -B /project/project_465002822 \
  -B /flash/project_465002822 \
  "${SIF}" \
  bash -c "PYTHONPATH=${PROJECT_DIR} python3 -u ${PROJECT_DIR}/tests/run_forward_only.py"
