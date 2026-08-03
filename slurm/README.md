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

## Normalisation arms

Four schemes, selected with `--normalization`. Run them as four jobs; each lands
in its own `outputs/<jobid>/` and records its scheme in `results.json`.

```bash
for scheme in zscore log1p_zscore minmax log1p_minmax; do
    sbatch --export=ALL,EXTRA_ARGS="--normalization $scheme" slurm/latent_comparison.slurm
done
```

`log1p_*` is **not** plain `log1p`. It is `sign(x) * log1p(|x| / s)` with `s` a
robust per-column scale, for two measured reasons: 60 of the 155 columns take
negative values, where `log1p` is undefined; and 26 columns have 99th-percentile
magnitudes below 0.1 in their native units, where `log1p` is the identity to
several decimal places. Cloud ice `QI` spans 7.7e-14 to 3.2e-6 kg/kg and plain
`log1p` leaves its dynamic range at 3.3 decades, unchanged. See
`src/greenland_art/autoencoder/normalization.py`.

`minmax` uses the true column min and max, so a single outlier sets the range.
That is deliberate — it is the honest behaviour of a linear rescale on columns
whose standardised extremes reach |z| = 448.

### Reading the results across arms

Explained variance ratio is **not comparable between schemes**: z-score gives
every column one unit of variance, min-max gives a heavy-tailed column almost
none. Use `physical_column_r2` instead, which inverts the normalisation and
scores each column in its original physical units — the one ruler outside all
four schemes.

Report its **median**, not its mean. The log arms reconstruct the typical column
slightly better than z-score while destroying a few: in a 60k-sample shakeout,
`log1p_zscore` PCA(15) scored median 0.879 against z-score's 0.863, but `RZ_L00`
came back at **-9526**. A small error in log space is a multiplicative error
after `expm1`, which is ruinous for a zero-inflated column with a wide range.
`physical_column_r2.columns_below_zero` is the number to watch.

### One confound to check, not assume

The MLP hyperparameters were chosen against z-scored inputs. Under min-max the
targets carry roughly 35x less variance, and the network starts much further
from them, so it needs more epochs to reach the same place. If a min-max arm
stops at `epochs_run == --epochs` rather than early-stopping, it was still
improving and its score understates the scheme rather than measuring it.

## Vista versus Lonestar6

Vista is **aarch64** (GH200); Lonestar6 GPU nodes are **x86_64** (A100). The
torch wheel differs between them, so let `uv` resolve on the machine you are
running on. Do not copy a `.venv` between the two — it will fail in a confusing
way at import time rather than at sync time.

## Shakeout first

UMAP dominates the wall clock, though far less than it used to — see below.
Confirm the rest of the pipeline runs before spending the full allocation:

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
- `reconstructions.npz` — full-map reconstruction examples in physical units:
  truth plus every PCA and autoencoder output for three whole days, with the
  validation membership of each cell so error can be quoted on held-out cells
  alone. Whole days rather than a random draw, because a reconstruction is
  judged on a map and a map needs every ice cell.
- `mlp_autoencoder_latent{15,30,100}.pt` — trained weights

- `normalization.npz`, `pca_latent*.npz` — the fitted scaling and PCA state

Copy the whole run directory back — it is ~25 MB. Nothing stores what the models
*output*: six full reconstructions of the matrix would be 14.6 GB, against 12 MB
for the six checkpoints, so reconstructions are recomputed locally on demand from
the checkpoints plus the training matrix (`greenland_art.autoencoder.SavedRun`).

### Error boxplots and the per-variable latent views

```bash
uv run python scripts/plot_error_summary.py --run-dir outputs/<jobid>
uv run python scripts/plot_latent_by_variable.py --run-dir outputs/<jobid>
```

The first writes `error_summary/<variable>/{timestep,year,day,year_month,month}.png`
— absolute error aggregated over the ice sheet at each time, six models as rows,
symlog y axis. 775 figures for all 155 variables; the first call caches
`error_statistics.npz`, which is the slow part. The second writes
`latent_by_variable/<variable>.png`, the PCA and UMAP views of every latent
width coloured by that variable's physical value, which shows which variables
the bottleneck organised itself by.

Both take `--fields` to render a subset.

### Interactive viewer

```bash
uv run bokeh serve --show scripts/reconstruction_explorer.py -- --run-dir outputs/<jobid>
```

Pick a model, a variable and a timestep; truth, prediction and |difference| are
computed live. This replaces rendering the 688,200 static frames those choices
span (103 GB). Colour limits are fixed over the whole decade rather than per
timestep, so scrubbing shows the field change and not the colour bar.

Copy the three `.json`/`.npz` files back; the figures are built locally with

```bash
uv run python scripts/plot_latent_comparison.py --run-dir outputs/<jobid>
```

which writes `latent_structure.png` (both projections of every latent width,
each coloured by season and by elevation), `latent_variance.png` and
`reconstruction_day<N>.png`. Use `--example-fields` and `--example-day` to
re-render other fields or either of the other two saved days.

## Why UMAP is no longer the bottleneck

Two changes, both measured on this hardware (14 cores):

**Embed a fixed subsample rather than fitting on one and transforming the rest
— 3.8x.** UMAP's `transform()` places unseen points into a layout that is
already frozen; it costs about as much as the fit and is lower fidelity than
letting those points take part in the optimisation. For a scatter plot there is
no reason to pay for it, and 300k points overplot into a solid blob anyway. The
same fixed subsample now drives every panel, which also makes them directly
comparable.

**Let UMAP use more than one thread — 7.4x.** `umap-learn` silently forces
`n_jobs=1` whenever `random_state` is set, because its layout optimisation is
order-dependent and therefore not reproducible in parallel. The only sign is a
warning most people never read. `--umap-parallel` gives up bit-identical output
for the parallelism, which is a reasonable trade for a 2D view and a bad one for
anything a number is read off. `results.json` records which mode was used.

On a node with many more cores the second factor should be larger still.

## If it is still too slow

RAPIDS `cuml.UMAP` runs the whole thing on the GPU and publishes **aarch64
wheels**, so it installs on Vista:

    uv pip install cuml-cu13

It is API-compatible enough to drop into `UMAPProjection`, but results are not
identical to `umap-learn` — different RNG and a different approximate-nearest-
neighbour backend. Untested here; there is no GPU on the machine this was
written on.
