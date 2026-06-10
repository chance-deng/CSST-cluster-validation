# Data Manifest

Large FITS catalogs and full intermediate products are intentionally not committed to git. The workflow expects the following local files or equivalent replacements.

## True Cluster Catalog

- `/Users/dengcanze/Documents/CSST/SMG_trace_ov_code/data/galaxy_clusters.fits`

## 7-band / LSTM Products

- `/Users/dengcanze/Documents/CSST/Codex/result/lstm_cross_hemisphere_photoz_i22/cross_runs_full/cross_hemisphere_lstm_predictions_combined.csv`
- `/Users/dengcanze/Documents/CSST/Codex/result/lstm_cross_hemisphere_photoz_i22/final_5field_catalogs/`
- `/Users/dengcanze/Documents/CSST/Codex/result/blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/`

## Blind-search Products

- `/Users/dengcanze/Documents/CSST/Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage/`
- `/Users/dengcanze/Documents/CSST/Codex/result/version_1_1_noppm_field05_7band_i_lt_22p0/`

## C6 Full True-galaxy Catalogs Used In Diagnostics

- `/Users/dengcanze/Documents/CSST/Codex/result/galaxies_C6_field05_blindsearch_input.fits`
- Server-side source example: `/data3/czdeng/CSST_overdensity/output_galaxy_catalog/galaxies_C6_all_bundles_all_types_unique.fits`

## Why These Are Excluded

Most FITS files are too large for a compact GitHub workflow repository. Summary CSVs and generated figures are included under `results/` and `figures/`.
