# COSMOS-Web DR1 Test With `blind_search.py`

This folder contains a fresh run of the cleaned `blind_search.py` on the COSMOS-Web DR1 photometric catalog.

## What Was Changed

Relative to the default `blind_search.py` / `config.yaml` settings, this run changes only the COSMOS-Web-specific inputs:

- input catalog: `/Users/dengcanze/Documents/SMG/cosmos_data/COSMOS-Web_DR1/filtered_galaxies_mask_all_easy.fits`
- FITS HDU: `1`
- columns: `ra`, `dec`, `zfinal`
- redshift range: `0.01 <= zfinal <= 3.7`
- quality selection: `flag_star == False`, `flag_star_hsc == 0`, and `warn_flag == 0`
- candidate-density background area: `1881.0 arcmin^2 = 0.522500 deg^2`
- tight-area definition: occupied fine-grid footprint with `cell=0.0125 deg`, computed after removing `flag_star_hsc != 0` sources. This avoids filling HSC star-mask holes, unlike the coarser CSST-field dilation mask.
- `delta_candidate` uses the final candidate's `n_member_scale_mpc`; the aperture is additionally clipped to the same fine-grid effective footprint to avoid overestimating the aperture area near field edges or HSC star-mask holes.

All core blind-search parameters are kept at the default values:

- redshift step: `0.01`
- slice half-width: `0.03(1+z)`
- pixel size: `0.3 arcmin`
- smoothing scales: `0.4, 0.8, 1.2 physical Mpc`
- background kernel factor: `15`
- raw peak cuts: `significance >= 0.2`, `delta >= 0.5`
- NMS: `1.0 Mpc/h`, `0.04(1+z)`, `min_detected_slices = 2`

## Outputs

- `cosmos_web_blind_search_config.yaml`
- `cosmos_web_dr1_our_blindsearch_candidates.csv`
- `cosmos_web_dr1_our_blindsearch_raw_peaks.csv`
- `cosmos_web_dr1_our_blindsearch_slice_stats.csv`
- `cosmos_web_dr1_our_blindsearch_merge_assignments.csv`
- `cosmos_web_dr1_our_blindsearch_summary.json`

## Run Summary

- selected galaxies: `541,637`
- redshift slices: `370`
- raw peaks: `41,292`
- final candidates: `1,718`

Top candidates:

| ID | RA | Dec | z_peak | significance | delta | delta_candidate | n_members | detected slices |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 149.932302 | 2.512732 | 0.70 | 4.560 | 8.353 | 0.523 | 1804 | 13 |
| 2 | 150.104907 | 1.964930 | 0.13 | 4.534 | 17.464 | -0.006 | 7844 | 9 |
| 3 | 149.936685 | 2.376562 | 0.73 | 4.162 | 7.472 | 0.454 | 1642 | 13 |
| 4 | 149.952920 | 2.351045 | 0.89 | 4.118 | 7.585 | 0.724 | 1388 | 15 |
| 5 | 150.091102 | 2.274003 | 0.71 | 4.105 | 7.121 | 0.313 | 1563 | 13 |

## Notes

The column `delta` is the Gaussian-smoothed local-background overdensity from the blind-search map. The column `delta_candidate` is the aperture richness overdensity proxy:

\[
\delta_{\rm candidate}
=
\frac{N_{\rm member}/[\pi(1.5R_G)^2]}
{N_{\rm slice,total}/S_{\rm tight}} - 1 .
\]

Here \(S_{\rm tight}=1881.0\,{\rm arcmin}^2\), computed from the HSC-star-masked input sample. The aperture scale follows `n_member_scale_mpc`, and the effective aperture area is clipped to the fine-grid footprint. Low-redshift candidates can still have high `n_members` but modest `delta_candidate` because the slice-wide background density is also high.
