# CSST Cluster Validation Workflow

This repository packages the current CSST cluster/proto-cluster validation workflow used for the competition report.

## Scope

The current package focuses on the completed cluster validation chain:

- 7-band cross-hemisphere LSTM photometric-redshift catalog.
- Field-level blind-search candidate detection.
- Tight field coverage definition for true-cluster denominators.
- Candidate-to-true-cluster cross-match validation.
- Recovery, purity proxy, false-detection proxy, candidate density, and threshold scans.
- Publication-style figures for coverage, photo-z, and 3D candidate distributions.

Large FITS catalogs are not committed. See [`DATA_MANIFEST.md`](DATA_MANIFEST.md) for local data paths and expected inputs.

## Main Report

Open the Chinese competition report here:

- [`docs/cluster_validation_report.md`](docs/cluster_validation_report.md)

All report figures are stored under [`figures/`](figures/) with relative Markdown links, so the report should render correctly in Obsidian and GitHub.

## Repository Layout

```text
.
├── docs/                 # Main competition report
├── figures/              # PNG figures referenced by reports
├── notebooks/            # Supplementary experiment notes
├── results/
│   ├── candidates/       # Compact candidate CSV tables
│   └── tables/           # Summary/metric CSV tables
├── src/                  # Python scripts for photo-z, blind search, cross-match, plotting
├── DATA_MANIFEST.md      # Large data products not tracked in git
└── REQUIREMENTS.md       # Python package notes
```


## Notes

This is a research workflow package rather than a pip-installable library. Paths in the original scripts are kept explicit to preserve provenance; adapt `PROJECT_ROOT` or input arguments before running on another machine.
