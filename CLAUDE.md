# greenland_art — project conventions

## Naming

**`snake_case`** for everything Python: variables, functions, modules, and file
names. `PascalCase` only for classes. Constants in `UPPER_SNAKE_CASE`.

Column names in DataFrames are also `snake_case`, including where that means
renaming a source column (`age_CE` from the NOAA files becomes `age_ce`).
Units go in the name when the value would otherwise be ambiguous: `x_km`,
`total_sulfate_ppb`, `age_top_bp`, `year_ce`.

Math code may use single letters (`a`, `e`, `phi`, `rho`) **only** where the
source equation is cited in the docstring — see
`src/greenland_art/visualization/projection.py`, which names Snyder (1987) and
maps every symbol.

## Data integrity

This is the rule that matters most here, because the project already lost work
to breaking it (see commit `4c19929`).

- **Never present generated data as observation.** No synthetic, stylised, or
  illustrative numbers in `datasets/` or `outputs/` without `synthetic_` in the
  filename *and* a statement on the figure itself.
- **Never invent place names, coordinates, or dates.** The removed code shipped
  four fictional glaciers.
- Every figure and app carries its sources and an explicit statement of whether
  the records are observational.
- Prefer a real record with an honest gap over a complete fabricated one. A
  site with no data in a window is drawn as an empty marker, not omitted and not
  interpolated.

## Data layout

`datasets/` is a symlink to `~/dataset/greenland_art`, outside the repo and
gitignored. Raw downloads live there untouched; nothing writes derived products
back into the raw source directories.

The volume backing it creates macOS AppleDouble `._name` sidecars. Any `glob`
over that tree must skip names starting with `._` — they match data globs but
are binary resource forks.

## Verification

Claims about interactivity get tested, not assumed. The exported preview is
checked by driving a real mouse drag through Chrome DevTools Protocol and
comparing the browser-computed values against the Python implementation
(`/tmp/test_brush.mjs` pattern). An HTML export that *looks* interactive but
has a dead callback is the same class of error as synthetic data.

## Visualisation

- Polar stereographic **EPSG:3413** for anything spatial. Never raw lon/lat on a
  Cartesian axis — Greenland spans 59–84 N and shears badly.
- Diverging blue↔red (`#2a78d6` / `#e34948`, neutral `#f0efec`) for anomalies;
  single-hue sequential for magnitude. Validated for colour-vision deficiency.
- Colour limits on comparable views are **fixed**, not rescaled per selection,
  so two selections can be compared.

## Environment

`uv` manages the environment. `uv run python scripts/<name>.py`. Do not
`pip install` into `.venv` directly.
