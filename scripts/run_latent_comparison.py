#!/usr/bin/env python3
"""Compare PCA, an MLP autoencoder, and UMAP on the MAR multi-field matrix.

Designed to run unattended on a GPU node. Writes a results JSON plus figures.

What each method is asked for, and why they are not scored on one axis:

  PCA     linear, deterministic, seconds to fit. The benchmark. Reported with
          per-component explained variance ratio, which is the only method here
          that decomposes variance component by component.
  MLP-AE  nonlinear. Reported with total explained variance ratio and MSE at
          matched latent width, so any gain over PCA is attributable to
          nonlinearity rather than to a wider bottleneck.
  UMAP    neighbour embedding for visualisation only. No reconstruction error is
          reported because it has no faithful inverse; a number there would
          invite a comparison that does not mean anything.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from greenland_art.autoencoder import (
    SCHEMES,
    MLPAutoencoder,
    Normalization,
    PCAModel,
    UMAPProjection,
)

DEFAULT_LATENT_DIMS = (15, 30, 100)


def physical_column_r2(model, validation, raw_validation, normalization) -> np.ndarray:
    """Per-column R^2 after mapping the reconstruction back to physical units.

    The explained variance ratio reported elsewhere is measured in whatever
    space the normalisation created, and those spaces are not comparable: z-score
    gives every column one unit of variance, min-max gives a heavy-tailed column
    almost none because one outlier sets its range. Ranking normalisation schemes
    against each other needs a ruler outside all of them, which is the original
    physical units of the data.
    """
    reconstruction = normalization.inverse_transform(model.reconstruct(validation))
    raw = raw_validation.astype(np.float64)
    residual = ((raw - reconstruction) ** 2).sum(axis=0)
    total = ((raw - raw.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - residual / np.where(total > 0.0, total, 1.0)


def reconstruction_sample_index(meta, target_days, requested_year=None):
    """Indices of every sample on a few whole days, so examples form full maps.

    Reconstruction quality is judged by eye on a map, and a map needs every ice
    cell. Taking whole days rather than a random draw is what makes that
    possible. The cost is that most of those cells were in training -- the split
    is random over samples, so roughly 90 % of any given day is -- which is why
    the validation membership travels with the arrays and the error figures
    quoted later are computed on that subset alone.
    """
    years = meta["year"]
    year = int(years.max()) if requested_year is None else requested_year
    available = np.unique(meta["day_of_year"][years == year])
    days = sorted({int(available[np.abs(available - target).argmin()]) for target in target_days})
    return np.flatnonzero((years == year) & np.isin(meta["day_of_year"], days)), year, days


def summarise_physical(scores: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "worst": float(np.min(scores)),
        "columns_below_zero": int((scores < 0.0).sum()),
    }


def load_matrix(path: Path, max_samples: int | None, seed: int):
    payload = np.load(path, allow_pickle=False)
    features = payload["features"]
    field_names = payload["field_names"]

    index = np.arange(len(features))
    if max_samples is not None and len(features) > max_samples:
        index = np.random.default_rng(seed).choice(len(features), max_samples, replace=False)
        index.sort()

    return features[index], field_names, {k: payload[k][index] for k in ("cell_index", "day_of_year", "year")}


def split(n_samples: int, validation_fraction: float, seed: int):
    shuffled = np.random.default_rng(seed).permutation(n_samples)
    cut = int(n_samples * validation_fraction)
    return shuffled[cut:], shuffled[:cut]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="npz from prepare_mar_training_data.py")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latent_comparison"))
    parser.add_argument("--latent-dims", type=int, nargs="+", default=list(DEFAULT_LATENT_DIMS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--umap-subsample", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalization", choices=list(SCHEMES), default="zscore",
        help="input scaling; see greenland_art.autoencoder.normalization",
    )
    parser.add_argument(
        "--reconstruction-days", type=int, nargs="+", default=[15, 195, 260],
        help="days of year to save full-map reconstruction examples for",
    )
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument(
        "--umap-parallel", action="store_true",
        help="Run UMAP multi-threaded. Roughly 7x faster on 14 cores, more on a "
             "larger node, at the cost of bit-identical reproducibility.",
    )
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    features, field_names, meta = load_matrix(arguments.input, arguments.max_samples, arguments.seed)
    n_samples, n_features = features.shape
    print(f"matrix {n_samples:,} x {n_features}", flush=True)

    usable = [d for d in arguments.latent_dims if d <= n_features]
    if len(usable) != len(arguments.latent_dims):
        dropped = sorted(set(arguments.latent_dims) - set(usable))
        print(f"dropping latent dims {dropped}: exceed n_features={n_features}", flush=True)

    train_index, validation_index = split(n_samples, 0.1, arguments.seed)
    normalization = Normalization.fit(features[train_index], arguments.normalization, arguments.seed)
    train = normalization.transform(features[train_index])
    raw_validation = features[validation_index]
    validation = normalization.transform(raw_validation)
    normalization.save(arguments.output_dir / "normalization.npz")
    print(
        f"train {len(train):,}  validation {len(validation):,}  "
        f"normalization {arguments.normalization}",
        flush=True,
    )

    results = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "field_names": [str(f) for f in field_names],
        "latent_dims": usable,
        "normalization": arguments.normalization,
        "umap_deterministic": not arguments.umap_parallel,
        "pca": {},
        "autoencoder": {},
    }
    latents = {}

    example_index, example_year, example_days = reconstruction_sample_index(
        meta, arguments.reconstruction_days
    )
    example_transformed = normalization.transform(features[example_index])
    examples = {
        "truth": features[example_index].astype(np.float32),
        "cell_index": meta["cell_index"][example_index],
        "day_of_year": meta["day_of_year"][example_index],
        "year": meta["year"][example_index],
        "is_validation": np.isin(example_index, validation_index),
    }
    print(
        f"reconstruction examples: {len(example_index):,} samples, "
        f"year {example_year}, days {example_days} "
        f"({int(examples['is_validation'].sum()):,} held out)",
        flush=True,
    )

    for latent_dim in usable:
        print(f"\n=== latent {latent_dim} ===", flush=True)

        started = time.time()
        pca = PCAModel(latent_dim, seed=arguments.seed).fit(train)
        ratios = pca.explained_variance_ratio_per_component
        pca_physical = physical_column_r2(pca, validation, raw_validation, normalization)
        results["pca"][str(latent_dim)] = {
            "explained_variance_ratio_per_component": [float(v) for v in ratios],
            "cumulative_explained_variance_ratio": [float(v) for v in np.cumsum(ratios)],
            "total_explained_variance_ratio": float(ratios.sum()),
            "validation_explained_variance_ratio": pca.explained_variance_ratio(validation),
            "validation_mse": pca.mean_squared_error(validation),
            "physical_column_r2": summarise_physical(pca_physical),
            "physical_column_r2_per_column": [float(v) for v in pca_physical],
            "fit_seconds": time.time() - started,
        }
        print(
            f"PCA({latent_dim}): total EVR {ratios.sum():.4f}  "
            f"val EVR {pca.explained_variance_ratio(validation):.4f}  "
            f"val MSE {pca.mean_squared_error(validation):.5f}  "
            f"physical col R2 mean {pca_physical.mean():.4f} "
            f"median {np.median(pca_physical):.4f} "
            f"(<0: {int((pca_physical < 0).sum())})",
            flush=True,
        )
        print(f"  first 10 component ratios: {np.round(ratios[:10], 4).tolist()}", flush=True)
        pca.save(arguments.output_dir / f"pca_latent{latent_dim}.npz")
        examples[f"pca{latent_dim}"] = normalization.inverse_transform(
            pca.reconstruct(example_transformed)
        ).astype(np.float32)

        started = time.time()
        autoencoder = MLPAutoencoder(
            latent_dim,
            hidden_sizes=tuple(arguments.hidden),
            max_epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            device=arguments.device,
            seed=arguments.seed,
        ).fit(train)
        autoencoder_physical = physical_column_r2(
            autoencoder, validation, raw_validation, normalization
        )
        results["autoencoder"][str(latent_dim)] = {
            "validation_explained_variance_ratio": autoencoder.explained_variance_ratio(validation),
            "validation_mse": autoencoder.mean_squared_error(validation),
            "physical_column_r2": summarise_physical(autoencoder_physical),
            "physical_column_r2_per_column": [float(v) for v in autoencoder_physical],
            "hidden_sizes": list(arguments.hidden),
            "epochs_run": len(autoencoder.history),
            "history": autoencoder.history,
            "fit_seconds": time.time() - started,
        }
        print(
            f"MLP-AE({latent_dim}): val EVR "
            f"{autoencoder.explained_variance_ratio(validation):.4f}  "
            f"val MSE {autoencoder.mean_squared_error(validation):.5f}  "
            f"physical col R2 mean {autoencoder_physical.mean():.4f} "
            f"median {np.median(autoencoder_physical):.4f} "
            f"(<0: {int((autoencoder_physical < 0).sum())})",
            flush=True,
        )
        autoencoder.save(arguments.output_dir / f"mlp_autoencoder_latent{latent_dim}.pt")
        latents[latent_dim] = autoencoder.encode(validation)
        examples[f"autoencoder{latent_dim}"] = normalization.inverse_transform(
            autoencoder.reconstruct(example_transformed)
        ).astype(np.float32)

    # One fixed subsample drives every 2D view. Three reasons to embed only
    # these rows rather than fit on a subsample and transform the rest: UMAP's
    # transform() costs about as much as the fit and is lower fidelity; a
    # 300k-point scatter overplots into a solid blob regardless; and sharing one
    # point set across panels makes them directly comparable.
    plot_count = min(arguments.umap_subsample, len(validation))
    plot_index = np.random.default_rng(arguments.seed).choice(
        len(validation), plot_count, replace=False
    )
    plot_index.sort()
    mode = "parallel, not reproducible" if arguments.umap_parallel else "single-thread, reproducible"
    print(
        f"\n2D views over {plot_count:,} of {len(validation):,} validation rows "
        f"(UMAP: {mode})", flush=True,
    )

    projections = {}
    views = [("input", validation[plot_index])]
    views += [(f"latent{d}", z[plot_index]) for d, z in latents.items()]

    for label, matrix in views:
        projections[f"{label}_pca2"] = PCAModel(2, seed=arguments.seed).fit_encode(matrix)
        if not arguments.skip_umap:
            started = time.time()
            projections[f"{label}_umap2"] = UMAPProjection(
                2, seed=arguments.seed, subsample=None,
                deterministic=not arguments.umap_parallel,
            ).fit_encode(matrix)
            print(f"UMAP on {label}: {time.time() - started:.0f}s", flush=True)

    np.savez_compressed(
        arguments.output_dir / "projections.npz",
        **{k: v.astype(np.float32) for k, v in projections.items()},
        plot_index=plot_index.astype(np.int32),
        validation_index=validation_index.astype(np.int32),
        cell_index=meta["cell_index"][validation_index][plot_index],
        day_of_year=meta["day_of_year"][validation_index][plot_index],
        year=meta["year"][validation_index][plot_index],
    )
    np.savez_compressed(
        arguments.output_dir / "reconstructions.npz",
        field_names=np.array([str(f) for f in field_names]),
        example_year=example_year,
        example_days=np.array(example_days),
        **examples,
    )
    with open(arguments.output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print(
        f"\nWrote {arguments.output_dir}/results.json, projections.npz "
        f"and reconstructions.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
