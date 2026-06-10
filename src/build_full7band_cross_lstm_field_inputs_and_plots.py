#!/usr/bin/env python3
"""Build full 7-band cross-LSTM field inputs with i-band magnitudes and plot each field."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.spatial import ConvexHull


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"
PRED_CSV = RESULT_ROOT / "lstm_cross_hemisphere_photoz/cross_runs_full/cross_hemisphere_lstm_predictions_combined.csv"
INPUT_DIR = RESULT_ROOT / "lstm_cross_hemisphere_photoz/inputs"
OUTROOT = RESULT_ROOT / "blindsearch_inputs/full7band_cross_lstm_i_mag_5fields"
NOTE = NOTEBOOK_ROOT / "full7band_cross_lstm_5field_distribution.md"
BANDS = ["NUV", "u", "g", "r", "i", "z", "y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTROOT)
    parser.add_argument("--note", type=Path, default=NOTE)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def mag_from_flux(values: pd.Series) -> np.ndarray:
    f = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(f), np.nan, dtype=np.float32)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = (25.0 - 2.5 * np.log10(f[ok])).astype(np.float32)
    return out


def magerr_from_flux(flux: pd.Series, err: pd.Series) -> np.ndarray:
    f = pd.to_numeric(flux, errors="coerce").to_numpy(float)
    e = pd.to_numeric(err, errors="coerce").to_numpy(float)
    out = np.full(len(f), np.nan, dtype=np.float32)
    ok = np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    out[ok] = (1.0857362047581294 * e[ok] / f[ok]).astype(np.float32)
    return out


def fixed_str(series: pd.Series) -> np.ndarray:
    arr = series.astype(str).to_numpy()
    max_len = max(1, max(len(x) for x in arr))
    return np.asarray(arr, dtype=f"U{max_len}")


def area_hull_bbox(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[ok]
    dec = dec[ok]
    if len(ra) < 3:
        return np.nan, np.nan
    dec0 = float(np.nanmedian(dec))
    bbox = float((np.nanmax(ra) - np.nanmin(ra)) * (np.nanmax(dec) - np.nanmin(dec)) * np.cos(np.radians(dec0)))
    x = (ra - np.nanmedian(ra)) * np.cos(np.radians(dec0))
    y = dec - np.nanmedian(dec)
    hull = float(ConvexHull(np.column_stack([x, y])).volume)
    return hull, bbox


def write_table(df: pd.DataFrame, out_path: Path) -> None:
    zfinal = pd.to_numeric(df["z_phot"], errors="coerce").to_numpy(np.float32)
    sigma = pd.to_numeric(df["z_mc_std"], errors="coerce").to_numpy(np.float32)
    fallback = (0.03 * (1.0 + zfinal)).astype(np.float32)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback).astype(np.float32)

    tab = Table()
    tab["row_idx"] = pd.to_numeric(df["row_idx"], errors="coerce").to_numpy(np.int64)
    tab["id"] = fixed_str(df["id"])
    tab["ra"] = pd.to_numeric(df["ra"], errors="coerce").to_numpy(np.float32)
    tab["dec"] = pd.to_numeric(df["dec"], errors="coerce").to_numpy(np.float32)
    tab["redshift"] = pd.to_numeric(df["redshift"], errors="coerce").to_numpy(np.float32)
    tab["z_true"] = pd.to_numeric(df["z_true"], errors="coerce").to_numpy(np.float32)
    tab["zfinal"] = zfinal
    tab["zpdf_l68"] = np.clip(zfinal - sigma, 0.0, 3.8).astype(np.float32)
    tab["zpdf_u68"] = np.clip(zfinal + sigma, 0.0, 3.8).astype(np.float32)
    tab["z_cross_lstm"] = zfinal
    tab["z_cross_lstm_sigma"] = sigma
    tab["z_mc_mean"] = pd.to_numeric(df["z_mc_mean"], errors="coerce").to_numpy(np.float32)
    tab["zConf"] = pd.to_numeric(df["zConf"], errors="coerce").to_numpy(np.float32)
    tab["mag_i"] = pd.to_numeric(df["mag_i"], errors="coerce").to_numpy(np.float32)
    tab["magerr_i"] = pd.to_numeric(df["magerr_i"], errors="coerce").to_numpy(np.float32)
    tab["n_valid_bands"] = pd.to_numeric(df["n_valid_bands"], errors="coerce").to_numpy(np.int16)
    tab["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").to_numpy(np.int16)
    tab["sky_half"] = fixed_str(df["sky_half"])
    for band in BANDS:
        tab[f"f_{band}"] = pd.to_numeric(df[f"f_{band}"], errors="coerce").to_numpy(np.float32)
        tab[f"e_{band}"] = pd.to_numeric(df[f"e_{band}"], errors="coerce").to_numpy(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tab.write(out_path, overwrite=True)


def plot_field(df: pd.DataFrame, field_id: int, hull: float, bbox: float, out_png: Path) -> None:
    setup_style()
    sample = df
    if len(sample) > 150_000:
        sample = sample.sample(150_000, random_state=42)
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    sc = ax.scatter(sample["ra"], sample["dec"], c=sample["mag_i"], s=0.25, alpha=0.45, linewidths=0, cmap="viridis_r", rasterized=True)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Field {field_id:02d}")
    ax.text(
        0.03,
        0.97,
        f"N = {len(df):,}\nHull area = {hull:.3f} deg$^2$\nBBox area = {bbox:.3f} deg$^2$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.78),
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$i$ mag")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    phot = []
    for half in ["hemisphere_A", "hemisphere_B"]:
        p = args.input_dir / f"csst_photometry_{half}.csv"
        df = pd.read_csv(p)
        df["mag_i"] = mag_from_flux(df["f_i"])
        df["magerr_i"] = magerr_from_flux(df["f_i"], df["e_i"])
        phot.append(df)
    phot_df = pd.concat(phot, ignore_index=True)

    keep_cols = [
        "row_idx",
        "id",
        "redshift",
        "n_valid_bands",
        "mag_i",
        "magerr_i",
        *[f"{prefix}_{band}" for band in BANDS for prefix in ("f", "e")],
    ]
    merged = pred.merge(phot_df[keep_cols], on=["row_idx", "id"], how="left", validate="one_to_one")
    if merged["mag_i"].isna().any():
        raise RuntimeError("Unmatched photometry rows when merging mag_i.")

    rows = []
    for field_id, part in merged.groupby("field_id"):
        field_id = int(field_id)
        half = str(part["sky_half"].iloc[0])
        out_fits = args.output_root / half / f"csst_field_{field_id:02d}_full7band_cross_lstm.fits"
        write_table(part, out_fits)
        ra = pd.to_numeric(part["ra"], errors="coerce").to_numpy(float)
        dec = pd.to_numeric(part["dec"], errors="coerce").to_numpy(float)
        hull, bbox = area_hull_bbox(ra, dec)
        out_png = plot_dir / f"field{field_id:02d}_full7band_cross_lstm_distribution.png"
        plot_field(part, field_id, hull, bbox, out_png)
        rows.append(
            {
                "field_id": field_id,
                "half": half,
                "rows": len(part),
                "area_hull_deg2": hull,
                "area_bbox_cosdec_deg2": bbox,
                "mag_i_median": float(np.nanmedian(pd.to_numeric(part["mag_i"], errors="coerce"))),
                "z_phot_median": float(np.nanmedian(pd.to_numeric(part["z_phot"], errors="coerce"))),
                "output_fits": str(out_fits),
                "plot_png": str(out_png),
            }
        )

    summary = pd.DataFrame(rows).sort_values("field_id").reset_index(drop=True)
    summary_csv = args.output_root / "full7band_cross_lstm_5field_summary.csv"
    summary.to_csv(summary_csv, index=False)

    lines = [
        "# Full 7band cross-LSTM five-field data distribution",
        "",
        f"- Prediction CSV: `{args.pred_csv}`",
        f"- Output root: `{args.output_root}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "| field | half | rows | hull_area_deg2 | bbox_area_deg2 | mag_i_median | z_phot_median | plot | FITS |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['field_id'])} | {row['half']} | {int(row['rows']):,} | {row['area_hull_deg2']:.3f} | {row['area_bbox_cosdec_deg2']:.3f} | {row['mag_i_median']:.3f} | {row['z_phot_median']:.3f} | `{row['plot_png']}` | `{row['output_fits']}` |"
        )
    lines += ["", "## Field plots", ""]
    for row in summary.to_dict(orient="records"):
        lines += [f"### Field {int(row['field_id']):02d}", "", f"![field{int(row['field_id']):02d}]({row['plot_png']})", ""]
    args.note.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote note: {args.note}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
