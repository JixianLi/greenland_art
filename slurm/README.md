# Running the latent comparison on TACC

## What gets transferred

Only two things: the prepared matrix and the repository. The raw MAR download
(19 GB of NetCDF per decade) stays local — `prepare_mar_training_data.py`
reduces it to a single `.npz`.

```bash
# locally, once
uv run python scripts/prepare_mar_training_data.py \
    --first-year 2000 --last-year 2009 --day-stride 5

# on the cluster: project at $SCRATCH/greenland_art, matrix inside its datasets/
ssh <user>@vista.tacc.utexas.edu "mkdir -p \$SCRATCH/greenland_art/datasets"
git clone git@github.com:JixianLi/greenland_art.git \$SCRATCH/greenland_art   # or git pull
scp datasets/mar/mar_training_2000_2009.npz \
    <user>@vista.tacc.utexas.edu:\$SCRATCH/greenland_art/datasets/
```

`datasets/` is a gitignored symlink locally, so the clone does not create it.
On the cluster it is a real directory holding the transferred matrix.

`--day-stride 5` keeps every fifth day. Stride 1 is ~9 GB, which is a painful
transfer for no obvious gain: 3.4 M samples is already far more than an MLP of
this size needs, and consecutive days are highly correlated anyway.

## Submitting

```bash
cd $SCRATCH/greenland_art
export SBATCH_ACCOUNT=<your-allocation>
sbatch --test-only slurm/latent_comparison.slurm   # confirm without queueing
sbatch slurm/latent_comparison.slurm
```

Defaults: `PROJECT_DIR=$SCRATCH/greenland_art`,
`DATA=$PROJECT_DIR/datasets/mar_training_2000_2009.npz`,
`OUTPUT_DIR=$PROJECT_DIR/outputs/<jobid>`. Override any of them with `--export`. The
script checks the matrix exists before doing anything else, so a wrong path
fails in seconds rather than after the environment resolves.

Three things in the script **must** be checked before the first submission —
they are placeholders or system-specific, and a wrong value either fails
immediately or silently queues forever:

| Line | What to set |
|---|---|
| allocation | Not in the script. `export SBATCH_ACCOUNT=<alloc>`, or `sbatch -A <alloc>`. `#SBATCH` lines do not expand variables. |
| `-p` | committed as `gh` (Vista). Use `gpu-a100` on Lonestar6, or `export SBATCH_PARTITION=...`. Confirm with `sinfo -s`. |
| `MODULES` | committed as `gcc/15.1.0 cuda/13.1` (Vista). Versions are pinned on purpose; bare `gcc cuda` takes the drifting site default. Override with `--export=ALL,MODULES="..."`. |

## Vista versus Lonestar6

Vista is **aarch64** (GH200); Lonestar6 GPU nodes are **x86_64** (A100). The
torch wheel differs between them, so let `uv` resolve on the machine you are
running on. Do not copy a `.venv` between the two — it will fail in a confusing
way at import time rather than at sync time.

## Shakeout first

UMAP dominates the wall clock. Confirm the rest of the pipeline runs before
spending the full allocation:

```bash
sbatch --export=ALL,EXTRA_ARGS="--max-samples 200000 --epochs 5 --skip-umap" \
       slurm/latent_comparison.slurm
```

## Outputs

Written to `$PROJECT_DIR/outputs/<jobid>/` — one directory per job, so
reruns do not overwrite each other and results stay separable from the
tracked figures already in `outputs/`:

- `results.json` — PCA per-component and cumulative explained variance ratio at
  each latent width, autoencoder validation EVR and MSE, and the full training
  history
- `projections.npz` — 2D PCA and 2D UMAP of the input and of each latent space
- `mlp_autoencoder_latent{15,30,100}.pt` — trained weights

Copy `results.json` and `projections.npz` back; the figures are built locally.
