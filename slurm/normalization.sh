for scheme in zscore log1p_zscore minmax log1p_minmax; do
    sbatch --export=ALL,EXTRA_ARGS="--normalization $scheme" slurm/latent_comparison.slurm
done
