#!/usr/bin/env python3
"""Run v1.1 blind-search only on field05 7band galaxies with an i-band magnitude cut."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
CODE_DIR = PROJECT_ROOT / "Codex/code"
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
BUILD_SCRIPT = CODE_DIR / "build_field05_7band_i_cut_blindsearch_input.py"


def label_from_limit(i_limit: float) -> str:
    return str(i_limit).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-limit", type=float, required=True)
    parser.add_argument("--force", action="store_true", help="Force blind-search rerun even if result CSV exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = label_from_limit(args.i_limit)
    input_fits = RESULT_ROOT / f"blindsearch_inputs/field05_7band_i_band_cuts/field05_7band_i_lt_{label}_blindsearch_input.fits"
    result_root = RESULT_ROOT / f"version_1_1_noppm_field05_7band_i_lt_{label}"

    if not input_fits.exists():
        subprocess.run([sys.executable, str(BUILD_SCRIPT), "--i-limit", str(args.i_limit)], check=True)

    sys.path.insert(0, str(CODE_DIR))
    import ppm_blindsearch_ppm_pipeline_v1_1 as pipe

    pipe.VERSION_TAG = f"version_1_1_noppm_field05_7band_i_lt_{label}"
    pipe.RESULT_ROOT = str(result_root)
    pipe.BLIND_OUTPUT_DIR = str(result_root / "blind_search")
    pipe.PPM_OUTPUT_DIR = str(result_root / "ppm_validation")
    pipe.WORKFLOW_OUTPUT_DIR = str(result_root / "workflow")
    pipe.LEGACY_RESULT_ROOT = str(result_root)
    pipe.LEGACY_BLIND_OUTPUT_DIR = str(result_root / "legacy_unused")
    pipe.MPL_CACHE_DIR = str(result_root / "mpl_cache")
    pipe.PY_CACHE_DIR = str(result_root / "__pycache__")
    pipe.REUSE_BLIND_RESULTS_IF_PRESENT = not args.force
    pipe.AUTO_BUILD_WORKFLOW = False

    os.environ["MPLCONFIGDIR"] = pipe.MPL_CACHE_DIR
    os.environ["PYTHONPYCACHEPREFIX"] = pipe.PY_CACHE_DIR
    os.environ["MPLBACKEND"] = "Agg"

    pipe.ensure_runtime_dirs()
    pipe.configure_core_paths()
    pipe.core.BLIND_INPUT_CATALOG = str(input_fits)
    pipe.core.PPM_BACKGROUND_CATALOG = str(input_fits)
    pipe.core.Z_MIN = 0.0
    pipe.core.Z_MAX = 2.2
    pipe.core.PPM_MAX_Z = 2.2

    print(f"Running v1.1 no-PPM blind search on field05 7band i<{args.i_limit:g}", flush=True)
    print(f"Input FITS: {input_fits}", flush=True)
    print(f"Result root: {result_root}", flush=True)

    blind_final_df, top_candidate_id, blind_csv_path = pipe.load_or_run_blind_search()
    pipe.dump_top_candidate_blind_density_slice(blind_final_df, top_candidate_id)

    summary_path = result_root / f"pipeline_summary_v1_1_noppm_field05_7band_i_lt_{label}.txt"
    top_row = blind_final_df[blind_final_df["ID"] == int(top_candidate_id)].iloc[0]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Version: {pipe.VERSION_TAG}\n")
        handle.write(f"Blind-search input: {pipe.core.BLIND_INPUT_CATALOG}\n")
        handle.write(f"Magnitude cut: mag_i < {args.i_limit:g}\n")
        handle.write(f"Blind-search candidates: {len(blind_final_df)}\n")
        handle.write(f"Top candidate ID: {top_candidate_id}\n")
        handle.write(f"Top candidate blind significance: {top_row['significance']:.3f}\n")
        handle.write(f"Blind-search CSV: {blind_csv_path}\n")
        handle.write("PPM stage: skipped\n")

    print(f"Finished i<{args.i_limit:g}: candidates={len(blind_final_df)}, csv={blind_csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
