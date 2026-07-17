# Blindsearch

A clean, configurable implementation of the **CSST blind overdensity cluster-finding algorithm**.

It detects galaxy-cluster candidates from photometric-redshift catalogs using **only sky coordinates (RA, Dec) and `z_phot`** (plus an optional magnitude cut). It does **not** rely on truth-cluster labels, halo information, red-sequence priors, PPM validation, or cross-match results during detection — all of those belong to post-processing validation, which is intentionally kept out of this repository.

The pipeline implements the fiducial algorithm described in the CSST cluster-search manuscript (C,Deng et al., in prep.):

- dynamic, overlapping redshift slices;
- Gaussian-smoothed count maps on a sky grid;
- a broad local-background kernel;
- overdensity $\delta$ and a Poisson-like ranking statistic $S$;
- greedy non-maximum-suppression (NMS) merging across redshift slices and smoothing scales;
- an aperture-based richness overdensity proxy $\delta_{\rm candidate}$.

> 这份代码是论文中正式描述的 blind-search 逻辑的纯净通用版本，所有关键参数都外置到 YAML 配置文件中，可直接用于不同输入星表、不同红移范围、不同平滑核和不同宇宙学参数。

---

## Table of Contents

- [Algorithm Overview](#algorithm-overview)
- [Environment & Dependencies](#environment--dependencies)
- [Quick Start](#quick-start)
- [Input Catalog Requirements](#input-catalog-requirements)
- [Configuration Reference (Step by Step)](#configuration-reference-step-by-step)
  - [1. `catalog`](#1-catalog)
  - [2. `cosmology`](#2-cosmology)
  - [3. `slices`](#3-slices)
  - [4. `density_map`](#4-density_map)
  - [5. `candidate_density`](#5-candidate_density)
  - [6. `peaks`](#6-peaks)
  - [7. `merge`](#7-merge)
  - [8. `output`](#8-output)
- [Running the Script](#running-the-script)
- [Output Files](#output-files)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Algorithm Overview

For each overlapping redshift slice centered at $z_s$, the slice membership rule is

$$
|z_{\rm phot}-z_s| \le 0.03\,(1+z_s).
$$

Galaxies in the slice are projected onto a 2D sky grid of pixel size $0.3'$ and smoothed with cluster-scale Gaussian kernels $R_G = 0.4,\,0.8,\,1.2$ physical Mpc. A broad local-background kernel ($15\times R_G$) is applied to the same count map. The overdensity and ranking statistic at pixel $(i,j)$ are

$$
\delta_{ij}=\frac{n^{\rm sm}_{ij}-n^{\rm bg}_{ij}}{n^{\rm bg}_{ij}},\qquad
S_{ij}=\frac{n^{\rm sm}_{ij}-n^{\rm bg}_{ij}}{\sqrt{n^{\rm bg}_{ij}}}.
$$

$S$ is used as an empirical ranking statistic for local overdensity peaks rather than as a full statistical likelihood, since it does not explicitly include cosmic variance or correlated large-scale structure. Raw peaks are identified as local maxima in the significance map using a $5\times5$ pixel maximum filter. A raw detection is retained if it satisfies $S_{ij}\ge 0.2$ and $\delta_{ij}\ge 0.5$. For each raw peak, the center is refined by averaging the positions of galaxies within $1.5R_G$ of the grid peak; the number of galaxies inside this aperture is recorded as $N_{\rm member}$.

To complement the Gaussian-smoothed $\delta_{ij}$, an aperture-based candidate overdensity is also evaluated:

$$
\delta_{\rm candidate}
=\frac{\rho_{\rm aper}-\rho_{\rm bg}}{\rho_{\rm bg}}
=\frac{N_{\rm member}}{\pi\,(fR_G)^2}\bigg/\frac{N_{\rm slice}}{S_{\rm tight}}-1,
$$

where $f=1.5$, $\rho_{\rm bg}=N_{\rm slice}/S_{\rm tight}$ is the mean surface density of the redshift slice converted to physical Mpc$^2$ using the angular-diameter distance, and the final candidate keeps the $\delta_{\rm candidate}$ from the constituent raw peak with the largest $N_{\rm member}$.

Raw peaks are then merged with greedy non-maximum suppression: two peaks are duplicates if their transverse separation is $\le 1.0$ Mpc/h **and** $|\Delta z|\le 0.04(1+z)$. A candidate is kept only if it is detected in at least 2 distinct redshift slices.

---

## Environment & Dependencies

The code is written in **Python 3.10+** and depends on:

| Package   | Purpose                                  |
|-----------|------------------------------------------|
| `numpy`   | array operations, histogramming          |
| `pandas`  | catalog I/O and tabular outputs          |
| `scipy`   | `scipy.ndimage.gaussian_filter`, `maximum_filter` |
| `astropy` | FITS/ECSV I/O, `FlatLambdaCDM` cosmology |
| `pyyaml`  | YAML config parsing (optional; JSON also supported) |

Install them with:

```bash
# using pip
pip install numpy pandas scipy astropy pyyaml

# or with conda
conda install numpy pandas scipy astropy pyyaml
```

Verify your environment:

```bash
python3 -c "import numpy, pandas, scipy, astropy; print('dependencies OK')"
```

---

## Quick Start

```bash
git clone <your-repo-url> blindsearch
cd blindsearch

# run with the bundled example config (edit catalog.path first)
python3 blind_search.py --config config.yaml
```

That single command will:

1. load the input catalog,
2. scan all redshift slices and extract raw overdensity peaks,
3. merge duplicate peaks with greedy NMS,
4. write all output products to the configured `output.directory`.

---

## Input Catalog Requirements

The pipeline needs three **mandatory** columns and up to two **optional** columns:

| Column              | Required | Unit    | Notes                                            |
|---------------------|:--------:|---------|--------------------------------------------------|
| RA                  | yes      | degree  | right ascension, named via `catalog.ra_column`   |
| Dec                 | yes      | degree  | declination, named via `catalog.dec_column`      |
| photometric redshift| yes      | —       | named via `catalog.redshift_column`              |
| magnitude           | no       | —       | only used if `magnitude_limit` is set            |

Supported formats (auto-detected by extension):

- `.fits` / `.fit` / `.fz` — read with `astropy.table.Table` (memmap)
- `.csv` / `.txt` — read with `pandas.read_csv`
- `.ecsv` — read with astropy

Rows with non-finite RA/Dec/z are dropped automatically. If a magnitude cut is configured, rows with non-finite magnitude or magnitude above the limit are also dropped.

---

## Configuration Reference (Step by Step)

All run-time parameters live in a single YAML (or JSON) file. A fully commented template is provided in [`config.yaml`](config.yaml). The sections below describe every configurable block. Each parameter shows its **default value** and the **fiducial value** used in the manuscript.

### 1. `catalog`

Input catalog path and column-name mapping. This is the only block you **must** edit for a new dataset.

```yaml
catalog:
  path: /path/to/your/catalog.fits
  ra_column: ra            # column name for RA in degrees
  dec_column: dec          # column name for Dec in degrees
  redshift_column: zfinal  # column name for photometric redshift
  magnitude_column: null   # set to a column name to enable a magnitude cut
  magnitude_limit: null    # keep rows with mag < this value (e.g. 22.0)
```

- `path` *(required)* — absolute or `~`-expanded path. Supports FITS / CSV / TXT / ECSV.
- `magnitude_column` + `magnitude_limit` — leave both `null` to disable the cut. Example: `i_band` + `22.0` keeps only `i < 22.0`.

> **Tip:** When switching to a new survey, usually only `path` and the three column names need to change.

### 2. `cosmology`

Flat $\Lambda$CDM parameters used to convert physical smoothing/merging scales into angular units.

```yaml
cosmology:
  h0: 67.66      # Hubble constant [km/s/Mpc]   (default: Planck 2018 / Jiutian)
  omega_m: 0.3111 # matter density              (default: Planck 2018 / Jiutian)
```

To reproduce the older `FlatLambdaCDM(H0=70, Om0=0.3)` test cosmology, set `h0: 70.0`, `omega_m: 0.3`.

### 3. `slices`

Redshift slicing controls the global search range and the overlapping slice width.

```yaml
slices:
  z_min: 0.0                      # global lower redshift bound   (fiducial 0.0)
  z_max: 2.2                      # global upper redshift bound   (fiducial 2.2)
  z_step: 0.01                    # slice-center step             (fiducial 0.01)
  half_width_factor: 0.03         # slice half-width = factor*(1+z) (fiducial 0.03)
  min_galaxies_per_slice: 5       # skip slices with fewer galaxies
```

Slice centers run from `z_min` to `z_max` in steps of `z_step`. A slice centered at $z_s$ keeps galaxies with $|z_{\rm phot}-z_s|\le$ `half_width_factor` $\times(1+z_s)$.

### 4. `density_map`

Sky-grid pixel size, Gaussian smoothing scales, and the local-background kernel.

```yaml
density_map:
  pixel_size_arcmin: 0.3          # sky pixel size [arcmin]        (fiducial 0.3)
  smoothing_scales_mpc: [0.4, 0.8, 1.2]  # R_G in physical Mpc     (fiducial)
  background_sigma_factor: 15.0   # background kernel = factor*R_G (fiducial 15)
  sigma_pixel_min: 1.0            # numerical clip on sigma_pix
  sigma_pixel_max: 15.0           # numerical clip on sigma_pix
  background_floor: 1.0e-5        # background lower bound (avoid divide-by-zero)
  maximum_filter_size: 5          # local-maximum window (5 = 5x5 pixels)
```

$\sigma_{\rm pix}=R_G\cdot s(z)/$`pixel_size` is clipped to `[sigma_pixel_min, sigma_pixel_max]` to avoid degenerate kernels at extreme redshifts.

### 5. `candidate_density`

Controls the aperture overdensity proxy $\delta_{\rm candidate}$.

```yaml
candidate_density:
  background_area_deg2: 8.364     # tight footprint area [deg^2]; null disables delta_candidate
  aperture_radius_factor: 1.5     # aperture radius = factor * R_G  (fiducial 1.5)
```

- `background_area_deg2` — the effective (tight) survey area. **Set to `null` to skip $\delta_{\rm candidate}$** (the column will be `NaN`).
- `aperture_radius_factor` ($f$) — aperture radius is $fR_G$.

Recall $\delta_{\rm candidate}$ depends strongly on the adopted $R_G$ (aperture area $\propto R_G^2$). For example, in a CSST Field-05 mock at $z=0.5$ with $R_G=0.4$ Mpc, one obtains roughly $\delta_{\rm candidate}\approx 4,\,12,\,20,\,25$ for $N_{\rm member}=20,\,50,\,80,\,100$ respectively.

### 6. `peaks`

Thresholds for retaining raw slice-level peaks.

```yaml
peaks:
  significance_min: 0.2      # keep peaks with S >= this   (fiducial 0.2)
  overdensity_min: 0.5       # keep peaks with delta >= this (fiducial 0.5)
  refinement_radius_factor: 1.5  # refine peak center using galaxies within factor*R_G
```

### 7. `merge`

Greedy NMS duplicate-merging rules and the persistence filter.

```yaml
merge:
  spatial_radius_mpc_h: 1.0      # spatial merge radius [physical Mpc/h] (fiducial 1.0)
  redshift_factor: 0.04          # merge if |dz| <= factor*(1+z)         (fiducial 0.04)
  min_detected_slices: 2         # require detection in >= this many slices (fiducial 2)
  z_bin: 0.02                    # coarse z index for NMS acceleration
  sky_cell_size_deg: 0.2         # coarse sky cell for NMS acceleration
```

`z_bin` and `sky_cell_size_deg` only affect NMS speed; they do not change the scientific merging definition.

### 8. `output`

Where to write results and what to name the files.

```yaml
output:
  directory: /path/to/output_dir   # required; created if missing
  candidate_filename: blindsearch_candidates.csv
  raw_peak_filename: blindsearch_raw_peaks.csv
  slice_filename: blindsearch_slice_stats.csv
  merge_filename: blindsearch_merge_assignments.csv
  summary_filename: blindsearch_summary.json
```

For GitHub-friendly repos you can use a relative path, e.g. `directory: results/my_run`.

---

## Running the Script

Once your config file is ready (say `config.yaml`), run:

```bash
python3 blind_search.py --config config.yaml
```

You can also point at any other config file without copying the script:

```bash
python3 /path/to/blind_search.py --config /path/to/my_config.yaml
```

The script prints live progress, e.g.:

```
Loaded 1,234,567 galaxies for blind search.
slice 1/221 z=0.000 dz=0.0300 galaxies=1234 peaks=0
slice 20/221 z=0.190 dz=0.0357 galaxies=5678 peaks=12
...
Final candidates: 1,718
Candidate table: /path/to/output_dir/blindsearch_candidates.csv
```

CLI help:

```bash
python3 blind_search.py --help
```

A typical end-to-end workflow:

1. **Prepare your catalog** in FITS/CSV with RA, Dec, and photo-z columns.
2. **Copy `config.yaml`** to `my_config.yaml`.
3. **Edit `catalog.path`** and the three column names; optionally set a magnitude cut.
4. **Set `output.directory`** to your results folder.
5. **Tune `slices` / `peaks` / `merge`** only if you are not using the fiducial values.
6. **Run** `python3 blind_search.py --config my_config.yaml`.
7. **Inspect** the candidate table and summary JSON.

---

## Output Files

Each run writes five files into `output.directory`:

| File                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `<candidate_filename>`     | Final NMS-merged candidate table (sorted by decreasing significance).      |
| `<raw_peak_filename>`      | All raw peaks, one row per (redshift slice, smoothing scale) detection.    |
| `<slice_filename>`         | Per-slice statistics: galaxy count, peak count, slice half-width.          |
| `<merge_filename>`         | Mapping from each raw peak to its final candidate (`seed` / `merged`).     |
| `<summary_filename>`       | Machine-readable run metadata: inputs, cosmology, counts, top candidate.   |

Key columns in the **candidate table**:

| Column               | Meaning                                                                                  |
|----------------------|------------------------------------------------------------------------------------------|
| `ID`                 | Final candidate identifier.                                                              |
| `RA`, `Dec`          | Refined peak center (degrees).                                                           |
| `z_peak`             | Redshift of the highest-significance seed slice.                                        |
| `significance`, `delta` | Gaussian-smoothed peak $S$ and $\delta$ at the seed.                                  |
| `delta_candidate`    | Aperture richness overdensity proxy (see formula above).                                |
| `best_scale_mpc`     | Largest smoothing scale that detected the candidate.                                     |
| `n_member_scale_mpc` | Scale at which the reported `N_member` was counted.                                     |
| `n_member_z_cen`     | Redshift at which the reported `N_member` was counted.                                  |
| `det_unique_slices`  | Number of distinct slices detecting the candidate (persistence).                        |
| `z_min_det`, `z_max_det`, `z_range` | Redshift extent of detections.                                            |
| `scales_hit`         | Pipe-separated list of smoothing scales that detected it.                               |
| `n_members`          | Maximum member count among constituent raw peaks.                                        |
| `n_slice_total`      | Total galaxies in the relevant slice.                                                   |
| `rho_aper_mpc2`, `rho_bg_mpc2`, `aperture_area_mpc2`, `background_area_mpc2` | Intermediate quantities for $\delta_{\rm candidate}$. |

---

## Examples

### COSMOS-Web DR1

[`examples/cosmos_web_dr1_z0p01_3p7_target1600/`](examples/cosmos_web_dr1_z0p01_3p7_target1600/) contains a COSMOS-Web DR1 run over $0.01\le z\le 3.7$, together with its config, candidate table, raw peaks, slice stats, merge assignments, and summary JSON. 

> **Note on extended keys:** the COSMOS-Web example YAML shows some extra keys (`hdu`, `exclude_flag_star`, `footprint_cell_deg`, etc.). These are consumed by an extended real-data finder; the clean `blind_search.py` in this repo ignores unknown keys and only reads the parameters documented under [Configuration Reference](#configuration-reference-step-by-step). Do not mix the COSMOS-Web `score`/`effective_members_peak` columns with the CSST-mock `delta_candidate`/`n_members` definitions.

---

## Project Structure

```
blindsearch/
├── blind_search.py                                  # main pipeline (single file, no package)
├── config.yaml                                      # fully commented example config
├── config_field05_i22_delta_candidate.yaml          # CSST Field-05 reference config
├── CONFIGURATION.md                                 # detailed configuration guide (bilingual)
├── COSMOS_WEB_CONFIGURATION.md                      # notes on the COSMOS-Web example
├── README.md                                        # this file
└── examples/
    └── cosmos_web_dr1_z0p01_3p7_target1600/
        ├── README.md
        ├── cosmos_web_blind_search_config.yaml
        ├── cosmos_web_dr1_our_blindsearch_candidates.csv
        ├── cosmos_web_dr1_our_blindsearch_raw_peaks.csv
        ├── cosmos_web_dr1_our_blindsearch_slice_stats.csv
        ├── cosmos_web_dr1_our_blindsearch_merge_assignments.csv
        └── cosmos_web_dr1_our_blindsearch_summary.json
```

For the full parameter-by-parameter guide (in bilingual Chinese/English), see [`CONFIGURATION.md`](CONFIGURATION.md).

---

## Citation

If you use this code in a publication, please cite the CSST cluster-search manuscript (C.Deng et al., in prep.) and link to this repository.

```bibtex
@ARTICLE{csst_blindsearch,
  title  = {Blindsearch: configurable blind overdensity cluster finder for CSST},
  author = {},
  url    = {<your-repo-url>},
  year   = {2026}
}
```

---

## License

This project is released for scientific use. See the repository license for details.
