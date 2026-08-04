cd $SCRATCH/greenland_art

for scheme in zscore log1p_zscore minmax log1p_minmax; do
    sbatch -J "latent-$scheme" --export=ALL,NORMALIZATION=$scheme \
           slurm/latent_comparison.slurm
done
