# Greenland Art Project

Exploring the Greenland Ice Sheet through observational data, with an eye toward
multi-sensory installation and AI-assisted analysis of multi-field records.

**Everything in `outputs/` is built from real observational records.** Nothing is
simulated, stylised, or synthetic. See `CLAUDE.md` for why this is stated so
prominently.

## Quick start

```bash
uv sync
uv run python scripts/download_preview_data.py    # ~3 MB, no account needed
uv run python scripts/build_preview_figure.py     # -> outputs/greenland_preview.png
uv run python scripts/build_preview_app.py        # -> outputs/greenland_preview.html
```

Open `outputs/greenland_preview.html` in any browser. It is self-contained: no
server, no Python, no network. Drag horizontally across the chemistry panel to
select a time window and the Greenland map recomputes.

## The preview

Two questions it is meant to answer.

**Can a timeline drive a spatial view?** Yes, and the linkage is real rather
than a mockup. Box-selecting a window on the GISP2 chemistry panel recomputes
each of the ten Osman core sites' δ¹⁸O anomaly for that window against its own
1750–1950 baseline, and repaints the map in EPSG:3413. The recomputation runs
client-side, and the browser's results have been checked against the Python
implementation to 4 decimal places.

**Is the multi-field data real and rich enough to be interesting?** Eight
chemical species are co-registered on a single time axis — the matrix an
autoencoder would consume. The industrial signal is unambiguous in it:

| GISP2 sulfate | Mean |
|---|---|
| 1550–1750 | 37.9 ppb |
| 1950–1988 | 110.7 ppb |

The volcanic sulfate panel is independent ground truth. The three largest
signals since 1700 are Laki (1783), Tambora (1815) and Katmai (1912) — none of
which were used to fit anything.

## Data

Downloaded by `scripts/download_preview_data.py`, all open HTTP, no credentials.

| Source | What | Coverage |
|---|---|---|
| **Osman et al. 2021** (PNAS) | δ¹⁸O + accumulation, 10 sites | annual, to 2013 |
| **GISP2 `chem/ionb.txt`** (Mayewski et al. 1997) | 8 major ions | ~2-yearly, to 1988 |
| **GISP2 `chem/volcano.txt`** (Zielinski et al. 1994) | total + volcanic sulfate | to 1985 |
| **GISP2 `physical/`** | accumulation, borehole temperature | — |
| **Natural Earth 50 m** | Greenland outline | — |

Osman sites span the full ice sheet: ACT2 (66.0 N) to Humboldt (78.5 N). Two are
accumulation-only, and Eurocore's δ¹⁸O ends in 1741 — the figures show these as
empty markers rather than hiding them.

### Known limits

- Greenland ice cores reach ~130 ka. They do **not** reach a million years; that
  needs marine sediment stacks (LR04, 5.3 Ma) or ice sheet model runs.
- The GRIP record's basal section is stratigraphically disturbed. Its nominal
  249 ka span should not be cited as clean climate signal.
- There is **no gridded chemistry product for Greenland**. Chemistry is
  point-source only. Gridded multi-field data means RACMO/MAR physical fields —
  see `docs/data_access.md`.
- ~69 chemistry samples cover the industrial era. That is a display target, not
  a training set.

## Layout

```
scripts/     download and build scripts
src/greenland_art/
  data/      loaders for the NOAA formats (osman2021, gisp2)
  analysis/  anomalies, standardisation, novelty scoring
  visualization/  EPSG:3413 projection, coastline, label placement
docs/        data_access.md — account setup for RACMO / MAR / ERA5
outputs/     generated figure and self-contained app
datasets/    symlink to ~/dataset/greenland_art (gitignored)
```

## Next

- Replace the placeholder novelty score with a trained autoencoder's
  reconstruction error; use latent trajectories as a second timeline.
- Bring in a real gridded field (RACMO 1 km, or ITS_LIVE velocity, which needs
  no account) so the spatial view is continuous rather than ten points.
- For the million-year direction: PISM and other open ice sheet models have
  published multi-million-year Greenland spin-ups, which is a smaller ask of a
  collaborator than a new simulation campaign.
