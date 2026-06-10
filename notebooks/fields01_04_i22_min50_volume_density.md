# Fields 01-04 i<22 Minimum-50 Candidate Volume Density Maps

- Candidate selection: highest integer `n_members` threshold that keeps at least 50 candidates.
- Density background: all full 7band galaxies in the corresponding field, no i-band cut.
- Gaussian smoothing: `sigma=0.2`.
- Red links connect matched candidates only when `R_proj <= 10 cMpc` and `Delta D <= 100 cMpc`.
- Summary CSV: `/Users/dengcanze/Documents/CSST/Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage/fields01_04_i22_min50_volume_density/fields01_04_i22_min50_volume_density_summary.csv`

| field | n_members >= | candidates | matched cand | unmatched cand | matched true clusters | match rate | purity proxy | linked pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 110 | 50 | 37 | 13 | 42 | 6.29% | 74.00% | 29 |
| 02 | 163 | 50 | 42 | 8 | 50 | 5.32% | 84.00% | 22 |
| 03 | 91 | 51 | 33 | 18 | 39 | 7.14% | 64.71% | 13 |
| 04 | 168 | 52 | 47 | 5 | 80 | 5.81% | 90.38% | 19 |

## Field 01

![field01](../figures/field01_i22_min50_volume_density_candidates.png)

## Field 02

![field02](../figures/field02_i22_min50_volume_density_candidates.png)

## Field 03

![field03](../figures/field03_i22_min50_volume_density_candidates.png)

## Field 04

![field04](../figures/field04_i22_min50_volume_density_candidates.png)
