#!/usr/bin/env python3
"""Run v1.1 no-PPM blind-search grid for fields 01-04 with i-band cuts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
CODE_DIR = PROJECT_ROOT / "Codex/code"
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
FIELD_ROOT = RESULT_ROOT / "blindsearch_inputs/full7band_cross_lstm_i_mag_5fields"
OUTROOT = RESULT_ROOT / "blindsearch_grid_full7band_fields01_04_i_cuts_v11"
NOTE = PROJECT_ROOT / "Codex/notebook/full7band_fields01_04_i_cut_blindsearch_grid.md"

I_LIMITS = [21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
NMEMBER_THRESHOLDS = [3, 5, 7, 10, 15, 20, 30, 50, 80, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", default="1,2,3,4")
    parser.add_argument("--i-limits", default=",".join(str(x) for x in I_LIMITS))
    parser.add_argument("--nmember-thresholds", default=",".join(str(x) for x in NMEMBER_THRESHOLDS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def label_i(value: float) -> str:
    return str(value).replace(".", "p")


def field_input_path(field_id: int) -> Path:
    half = "hemisphere_A" if field_id in {1, 3, 4} else "hemisphere_B"
    return FIELD_ROOT / half / f"csst_field_{field_id:02d}_full7band_cross_lstm.fits"


def build_i_cut_input(field_id: int, i_limit: float) -> tuple[Path, int, int]:
    src = field_input_path(field_id)
    out_dir = OUTROOT / "inputs" / f"field{field_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"csst_field_{field_id:02d}_full7band_i_lt_{label_i(i_limit)}_blindsearch_input.fits"
    if out.exists():
        tab = Table.read(out, memmap=True)
        return out, -1, len(tab)

    tab = Table.read(src, memmap=True)
    mag_i = np.asarray(tab["mag_i"], dtype=float)
    z = np.asarray(tab["zfinal"], dtype=float)
    keep = np.isfinite(mag_i) & (mag_i < i_limit) & np.isfinite(z) & (z >= 0.0) & (z <= 2.2)
    out_tab = tab[keep]
    out_tab.write(out, overwrite=True)
    return out, len(tab), len(out_tab)


def run_blindsearch(field_id: int, i_limit: float, input_fits: Path, force: bool) -> tuple[pd.DataFrame, Path]:
    sys.path.insert(0, str(CODE_DIR))
    import ppm_blindsearch_ppm_pipeline_v1_1 as pipe

    label = label_i(i_limit)
    result_root = OUTROOT / f"field{field_id:02d}_i_lt_{label}"
    pipe.VERSION_TAG = f"v1_1_noppm_field{field_id:02d}_full7band_i_lt_{label}"
    pipe.RESULT_ROOT = str(result_root)
    pipe.BLIND_OUTPUT_DIR = str(result_root / "blind_search")
    pipe.PPM_OUTPUT_DIR = str(result_root / "ppm_validation")
    pipe.WORKFLOW_OUTPUT_DIR = str(result_root / "workflow")
    pipe.LEGACY_RESULT_ROOT = str(result_root)
    pipe.LEGACY_BLIND_OUTPUT_DIR = str(result_root / "legacy_unused")
    pipe.MPL_CACHE_DIR = str(result_root / "mpl_cache")
    pipe.PY_CACHE_DIR = str(result_root / "__pycache__")
    pipe.REUSE_BLIND_RESULTS_IF_PRESENT = not force
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

    blind_final_df, top_candidate_id, blind_csv_path = pipe.load_or_run_blind_search()
    pipe.dump_top_candidate_blind_density_slice(blind_final_df, top_candidate_id)

    top_row = blind_final_df[blind_final_df["ID"] == int(top_candidate_id)].iloc[0]
    summary_path = result_root / f"pipeline_summary_field{field_id:02d}_i_lt_{label}.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Version: {pipe.VERSION_TAG}\n")
        handle.write(f"Blind-search input: {pipe.core.BLIND_INPUT_CATALOG}\n")
        handle.write(f"Magnitude cut: mag_i < {i_limit:g}\n")
        handle.write(f"Blind-search candidates: {len(blind_final_df)}\n")
        handle.write(f"Top candidate ID: {top_candidate_id}\n")
        handle.write(f"Top candidate blind significance: {top_row['significance']:.3f}\n")
        handle.write(f"Blind-search CSV: {blind_csv_path}\n")
        handle.write("PPM stage: skipped\n")
    return blind_final_df, Path(blind_csv_path)


def main() -> int:
    args = parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    fields = [int(x.strip()) for x in args.fields.split(",") if x.strip()]
    i_limits = [float(x.strip()) for x in args.i_limits.split(",") if x.strip()]
    nmember_thresholds = [int(x.strip()) for x in args.nmember_thresholds.split(",") if x.strip()]

    rows = []
    threshold_rows = []
    total = len(fields) * len(i_limits)
    done = 0
    for field_id in fields:
        for i_limit in i_limits:
            done += 1
            print(f"\n[{done}/{total}] field {field_id:02d}, i<{i_limit:g}", flush=True)
            input_fits, rows_total, rows_kept = build_i_cut_input(field_id, i_limit)
            print(f"Input FITS: {input_fits}, rows_kept={rows_kept:,}", flush=True)
            blind_df, blind_csv = run_blindsearch(field_id, i_limit, input_fits, args.force)
            top = blind_df.iloc[0]
            rows.append(
                {
                    "field_id": field_id,
                    "i_limit": i_limit,
                    "input_rows_total": rows_total,
                    "input_rows_kept": rows_kept,
                    "candidate_count": len(blind_df),
                    "top_id": int(top["ID"]),
                    "top_significance": float(top["significance"]),
                    "top_n_members": int(top["n_members"]),
                    "blind_csv": str(blind_csv),
                    "input_fits": str(input_fits),
                }
            )
            for thr in nmember_thresholds:
                kept = blind_df[pd.to_numeric(blind_df["n_members"], errors="coerce") >= thr]
                threshold_rows.append(
                    {
                        "field_id": field_id,
                        "i_limit": i_limit,
                        "n_members_threshold": thr,
                        "candidate_count_after_threshold": len(kept),
                    }
                )
            pd.DataFrame(rows).to_csv(OUTROOT / "blindsearch_i_cut_grid_summary.csv", index=False)
            pd.DataFrame(threshold_rows).to_csv(OUTROOT / "blindsearch_i_cut_nmember_threshold_summary.csv", index=False)

    summary = pd.DataFrame(rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    summary_csv = OUTROOT / "blindsearch_i_cut_grid_summary.csv"
    threshold_csv = OUTROOT / "blindsearch_i_cut_nmember_threshold_summary.csv"
    summary.to_csv(summary_csv, index=False)
    threshold_summary.to_csv(threshold_csv, index=False)

    def pivot_to_md(pivot: pd.DataFrame) -> str:
        frame = pivot.reset_index()
        cols = [str(c) for c in frame.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---:"] * len(cols)) + " |",
        ]
        for _, rec in frame.iterrows():
            vals = []
            for col in frame.columns:
                value = rec[col]
                if pd.isna(value):
                    vals.append("")
                elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                    vals.append(f"{int(value):,}")
                elif isinstance(value, (int, np.integer)):
                    vals.append(f"{int(value):,}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Full 7band fields 01-04 i-band cut blind-search grid",
        "",
        f"- Output root: `{OUTROOT}`",
        f"- Grid summary: `{summary_csv}`",
        f"- n_members threshold summary: `{threshold_csv}`",
        "",
        "## Blind-search grid",
        "",
        "| field | i_limit | input_rows | candidates | top_significance | top_n_members |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['field_id'])} | {row['i_limit']:.1f} | {int(row['input_rows_kept']):,} | {int(row['candidate_count']):,} | {row['top_significance']:.3f} | {int(row['top_n_members'])} |"
        )
    lines += [
        "",
        "## n_members threshold scan",
        "",
        "该表只统计每个 blind-search candidate CSV 中经过 `n_members >= threshold` 后剩余的候选体数量；尚未在这里做 true-cluster cross-match。",
        "",
    ]
    for field_id in fields:
        lines += [f"### Field {field_id:02d}", ""]
        part = threshold_summary[threshold_summary["field_id"] == field_id]
        pivot = part.pivot(index="n_members_threshold", columns="i_limit", values="candidate_count_after_threshold")
        lines.append(pivot_to_md(pivot))
        lines.append("")
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote note: {NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
