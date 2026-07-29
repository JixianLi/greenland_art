# Getting access to gridded Greenland data (RACMO, MAR, ERA5)

The preview in `outputs/` runs entirely on open NOAA ice core data with no
account. This document covers the next tier: **gridded, multi-field,
spatiotemporal** model output, which is what the multi-field AI direction
actually needs. Some of it requires registration.

Status checked 2026-07-29. Endpoint reachability was tested from this machine;
where a source was unreachable it says so rather than pretending otherwise.

---

## Why we need this tier at all

The ice cores give us many chemical fields at *one point* (GISP2) or two fields
at *ten points* (Osman). Neither is a spatial field. There is **no gridded
chemistry product for Greenland** — chemistry only exists as core records.

RACMO and MAR give the opposite trade: a dense grid, many *physical* fields
(surface mass balance, melt, runoff, snowfall, near-surface temperature,
albedo, wind), 1958–present, at 5.5 km / 1 km — but no chemistry.

For training anything, this matters a lot: the GISP2 industrial-era window is
~69 samples. RACMO/MAR is millions of spatial samples per field per year. The
ice cores are a **display** target; the gridded model output is a **training**
target.

---

## 1. Copernicus CDS — ERA5 (do this one first)

ERA5 is the atmospheric reanalysis, 1940–present, and it is also what forces
recent MAR and RACMO runs. Most useful on its own as a temperature/precipitation
field over Greenland.

- Portal: <https://cds.climate.copernicus.eu/> (verified reachable, HTTP 200)
- Since the 2024 CDS migration you need an **ECMWF account**, not the old CDS
  login. Register at <https://www.ecmwf.int/> and then log into CDS with it.

After registering:

1. Log in, open your profile page, copy your **API token**.
2. Write `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-token>
   ```
   `chmod 600 ~/.cdsapirc`
3. **Accept the licence for each dataset in the web UI before any API request.**
   This is the step that catches everyone — the API returns an authorisation
   error, not a licence error, so it reads like a broken key.
4. `uv add cdsapi` when we are ready to script downloads.

Datasets worth pulling for Greenland:
- `reanalysis-era5-single-levels` — 2 m temperature, total precipitation, snowfall
- `reanalysis-era5-land` — 9 km land surface, higher resolution

---

## 2. MAR (Modèle Atmosphérique Régional) — Université de Liège

MAR is Xavier Fettweis' regional climate model; the Greenland runs are the
standard reference for surface melt and SMB.

- The usual bulk host is `ftp.climato.be`. **This was unreachable from this
  machine** (connection timed out on both `/fettweis/` and `/climato/`). It may
  be firewalled here, temporarily down, or moved — worth trying from your
  network before assuming it is gone.
- **Zenodo is the reliable fallback and needs no account.** Verified records:

  | Zenodo ID | DOI | What |
  |---|---|---|
  | 5024965 | `10.5281/zenodo.5024965` | Fettweis et al. 2021 (The Cryosphere) — Greenland ice sheet and geoengineering, MAR outputs |
  | 10066185 | `10.5281/zenodo.10066185` | MAR coupled to an ice sheet model, Greenland |
  | 15386343 | `10.5281/zenodo.15386343` | Paice et al. — feedbacks driving Greenland evolution |

- For a run not on Zenodo, emailing Fettweis directly is the normal and accepted
  route; MAR output is shared on request for research use.

---

## 3. RACMO2 — IMAU, Utrecht University

RACMO2.3p2 / 2.4p1 is the other standard Greenland SMB model, including the
statistically downscaled 1 km product.

- The IMAU project page URL I tried (`projects.science.uu.nl/iceclimate/...`)
  returned **404** — that page has moved. Do not rely on it.
- **Zenodo again works and needs no account.** Verified records:

  | Zenodo ID | DOI | What |
  |---|---|---|
  | 3367211 | `10.5281/zenodo.3367211` | Downscaled **1 km** RACMO2 Greenland SMB |
  | 3368405 | `10.5281/zenodo.3368405` | **11 km** RACMO2 Greenland |
  | 4289959 | `10.5281/zenodo.4289959` | 21st-century warming threshold for sustained Greenland mass loss |
  | 7100706 | `10.5281/zenodo.7100706` | Peak refreezing in the Greenland firn layer |

- Full-resolution, full-period RACMO is generally **by request to IMAU** (Brice
  Noël / Michiel van den Broeke). Institutional email helps; state the project
  and intended use. Expect days, not minutes.

---

## 4. NASA Earthdata — already configured

Your credentials are at `~/.netrc` (mode 600, `machine urs.earthdata.nasa.gov`).
Note this is `.netrc`, **not** `.ncrt` — tools look for `.netrc` and that is
where yours already is, so nothing to do.

This unlocks:
- **NSIDC** — BedMachine Greenland v5 (bed topography, ice thickness, surface),
  MEaSUREs ice velocity. <https://nsidc.org/data/idbmg4/versions/5> (verified 200)
- **GRACE / GRACE-FO** mascons via PO.DAAC — ice sheet mass change, 2002–present

Some NSIDC products need the dataset-specific "study" authorisation approved in
your Earthdata profile even with valid credentials.

---

## 5. ITS_LIVE — no account at all

Worth knowing about because it is the least friction of anything on this page.
Annual ice velocity mosaics, 1985–present, in Zarr on a public S3 bucket.
Anonymous listing verified working (HTTP 200):

```
https://its-live-data.s3.amazonaws.com/?list-type=2&prefix=velocity_mosaic/
```

No credentials, no licence click-through. If we want a real gridded spatial
field on screen quickly, this is the fastest path.

---

## Suggested order

1. **ERA5 / Copernicus** — needed for the most things, slowest to set up.
2. **Zenodo RACMO 1 km** (`10.5281/zenodo.3367211`) — no account, gives us a real
   multi-field grid to prototype the autoencoder on immediately.
3. **ITS_LIVE** — no account, adds ice dynamics.
4. **Request full RACMO/MAR from IMAU / ULiège** — start the email early, it has
   the longest lead time.
