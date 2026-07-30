# Running the latent comparison on TACC

## What gets transferred

Only two things: the prepared matrix and the repository. The raw MAR download
(19 GB of NetCDF per decade) stays local — `prepare_mar_training_data.py`
reduces it to a single `.npz`.

```bash
# locally, once
uv run python scripts/prepare_mar_training_data.py \
    --first-year 2000 --last-year 2009 --day-stride 5

scp datasets/mar/mar_training_2000_2009.npz  <user>@ls6.tacc.utexas.edu:\$SCRATCH/
rsync -av --exclude .venv --exclude datasets --exclude outputs \
    ./ <user>@ls6.tacc.utexas.edu:\$WORK/greenland_art/
```

`--day-stride 5` keeps every fifth day. Stride 1 is ~9 GB, which is a painful
transfer for no obvious gain: 3.4 M samples is already far more than an MLP of
this size needs, and consecutive days are highly correlated anyway.

## Submitting

```bash
cd $WORK/greenland_art
sbatch --export=ALL,DATA=$SCRATCH/mar_training_2000_2009.npz \
       slurm/latent_comparison.slurm
```

Three things in the script **must** be checked before the first submission —
they are placeholders or system-specific, and a wrong value either fails
immediately or silently queues forever:

| Line | What to set |
|---|---|
| `-A` | your allocation. The job will not start without it. |
| `-p` | `gpu-a100` on Lonestar6, `gh` on Vista. Confirm with `sinfo -s`. |
| `module load` | TACC module names differ per system and change over time. |

## Vista versus Lonestar6

Vista is **aarch64** (GH200); Lonestar6 GPU nodes are **x86_64** (A100). The
torch wheel differs between them, so let `uv` resolve on the machine you are
running on. Do not copy a `.venv` between the two — it will fail in a confusing
way at import time rather than at sync time.

## Shakeout first

UMAP dominates the wall clock. Confirm the rest of the pipeline runs before
spending the full allocation:

```bash
sbatch --export=ALL,DATA=$SCRATCH/mar_training_2000_2009.npz,\
EXTRA_ARGS="--max-samples 200000 --epochs 5 --skip-umap" \
       slurm/latent_comparison.slurm
```

## Outputs

Written to `$SCRATCH/greenland_latent/<jobid>/`:

- `results.json` — PCA per-component and cumulative explained variance ratio at
  each latent width, autoencoder validation EVR and MSE, and the full training
  history
- `projections.npz` — 2D PCA and 2D UMAP of the input and of each latent space
- `mlp_autoencoder_latent{15,30,100}.pt` — trained weights

Copy `results.json` and `projections.npz` back; the figures are built locally.
