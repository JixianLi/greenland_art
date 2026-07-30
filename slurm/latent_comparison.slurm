#!/bin/bash
#----------------------------------------------------------------------------
# PCA / MLP-autoencoder / UMAP comparison on the MAR multi-field matrix.
#
# Written for TACC. Defaults target Lonestar6 A100 nodes; see the Vista block
# below for the GH200 partition names.
#
#   sbatch --export=ALL,DATA=$SCRATCH/mar_training_2000_2009.npz latent_comparison.slurm
#
# VERIFY BEFORE SUBMITTING:
#   -A            your allocation name (this will not run without it)
#   -p            partition; check `sinfo -s` on the actual system
#   the module load line -- TACC module names change between systems
#----------------------------------------------------------------------------
#SBATCH -J greenland-latent
#SBATCH -o logs/latent_%j.out
#SBATCH -e logs/latent_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -p gpu-a100
#SBATCH -A REPLACE_WITH_ALLOCATION
#SBATCH --mail-type=END,FAIL

# Vista (GH200) instead of Lonestar6: use -p gh (or gh-dev for short tests).
# Vista is aarch64, so the uv-resolved torch wheel differs from Lonestar6's
# x86_64 one. Resolve the environment on the machine you intend to run on --
# do not copy .venv between them.

set -euo pipefail

module purge
module load gcc cuda || echo "module load failed; relying on the uv environment"

PROJECT_DIR="${PROJECT_DIR:-$WORK/greenland_art}"
DATA="${DATA:-$SCRATCH/mar_training_2000_2009.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/greenland_latent/$SLURM_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "host       : $(hostname)"
echo "job        : ${SLURM_JOB_ID:-none}"
echo "data       : $DATA"
echo "output     : $OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"

# uv resolves and caches into the project .venv on first run. Keep the cache on
# a filesystem the compute node can actually write to; $HOME is small on TACC
# and the default cache location will fill it.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
uv sync --frozen || uv sync

# --max-samples caps the matrix for a quick shakeout run; drop it for the full
# fit. UMAP is the slow stage by far, so it is worth confirming the rest works
# with --skip-umap before spending the full walltime.
uv run python scripts/run_latent_comparison.py \
    --input "$DATA" \
    --output-dir "$OUTPUT_DIR" \
    --latent-dims 15 30 100 \
    --epochs 120 \
    --batch-size 8192 \
    --hidden 512 256 128 \
    --umap-subsample 150000 \
    --device cuda \
    "${EXTRA_ARGS:-}"

echo "done: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
