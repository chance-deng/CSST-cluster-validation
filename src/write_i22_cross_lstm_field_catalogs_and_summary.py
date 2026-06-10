#!/usr/bin/env python3
"""Write i<22 cross-hemisphere LSTM field catalogs, plots, and summary note."""

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

PRED_CSV = RESULT_ROOT / "lstm_cross_hemisphere_photoz_i22/cross_runs_full/cross_hemisphere_lstm_predictions_combined.csv"
INPUT_DIR = RESULT_ROOT / "lstm_cross_hemisphere_photoz_i22/inputs"
METRICS_CSV = RESULT_ROOT / "lstm_cross_hemisphere_photoz_i22/cross_runs_full/cross_hemisphere_lstm_metrics.csv"
OUTROOT = RESULT_ROOT / "lstm_cross_hemisphere_photoz_i22/final_5field_catalogs"
NOTE_PATH = NOTEBOOK_ROOT / "csst_7band_i22_cross_hemisphere_lstm_summary.md"

BANDS = ["NUV", "u", "g", "r", "i", "z", "y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--output-root", type=Path, default=OUTROOT)
    parser.add_argument("--note-path", type=Path, default=NOTE_PATH)
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


def area_deg2(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[ok]
    dec = dec[ok]
    if len(ra) < 3:
        return np.nan, np.nan
    bbox = (float(np.nanmax(ra) - np.nanmin(ra)) * float(np.nanmax(dec) - np.nanmin(dec)) * np.cos(np.radians(float(np.nanmedian(dec)))))
    x = (ra - np.nanmedian(ra)) * np.cos(np.radians(float(np.nanmedian(dec))))
    y = dec - np.nanmedian(dec)
    hull = float(ConvexHull(np.column_stack([x, y])).volume)
    return hull, bbox


def flux_to_mag(flux: pd.Series) -> np.ndarray:
    f = pd.to_numeric(flux, errors="coerce").to_numpy(float)
    mag = np.full(len(f), np.nan, dtype=np.float32)
    ok = np.isfinite(f) & (f > 0)
    mag[ok] = (25.0 - 2.5 * np.log10(f[ok])).astype(np.float32)
    return mag


def fluxerr_to_magerr(flux: pd.Series, err: pd.Series) -> np.ndarray:
    f = pd.to_numeric(flux, errors="coerce").to_numpy(float)
    e = pd.to_numeric(err, errors="coerce").to_numpy(float)
    out = np.full(len(f), np.nan, dtype=np.float32)
    ok = np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    out[ok] = (1.0857362047581294 * e[ok] / f[ok]).astype(np.float32)
    return out


def load_inputs(input_dir: Path) -> pd.DataFrame:
    frames = []
    for half in ["hemisphere_A", "hemisphere_B"]:
        p = input_dir / f"csst_photometry_{half}_i_lt_22.csv"
        df = pd.read_csv(p)
        if "mag_i" not in df.columns:
            df["mag_i"] = flux_to_mag(df["f_i"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metrics(z_true: np.ndarray, z_phot: np.ndarray) -> dict[str, float | int]:
    ok = np.isfinite(z_true) & np.isfinite(z_phot) & (z_true > 0)
    dz = (z_phot[ok] - z_true[ok]) / (1.0 + z_true[ok])
    if dz.size == 0:
        return {"N": 0, "sigma_NMAD": np.nan, "outlier_fraction": np.nan, "bias": np.nan}
    bias = float(np.median(dz))
    return {
        "N": int(dz.size),
        "sigma_NMAD": float(1.48 * np.median(np.abs(dz - bias))),
        "outlier_fraction": float(np.mean(np.abs(dz) > 0.15)),
        "bias": bias,
    }


def plot_photoz(df: pd.DataFrame, half: str, out_png: Path) -> None:
    setup_style()
    z_true = pd.to_numeric(df["z_true"], errors="coerce").to_numpy(float)
    z_phot = pd.to_numeric(df["z_phot"], errors="coerce").to_numpy(float)
    stat = metrics(z_true, z_phot)
    ok = np.isfinite(z_true) & np.isfinite(z_phot)
    rng = np.random.default_rng(42)
    idx = np.where(ok)[0]
    if len(idx) > 300_000:
        idx = rng.choice(idx, 300_000, replace=False)

    fig, ax = plt.subplots(figsize=(4.3, 4.1))
    ax.scatter(z_true[idx], z_phot[idx], s=1.0, c="black", alpha=0.08, linewidths=0, rasterized=True)
    xx = np.linspace(0, 2.2, 300)
    ax.plot(xx, xx, color="#B2182B", lw=1.1)
    ax.plot(xx, xx + 0.15 * (1 + xx), color="#2166AC", lw=0.8, ls="--")
    ax.plot(xx, xx - 0.15 * (1 + xx), color="#2166AC", lw=0.8, ls="--")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(r"$z_{\rm true}$")
    ax.set_ylabel(r"$z_{\rm phot}$")
    ax.set_title(f"{half} cross-target")
    ax.text(
        0.04,
        0.96,
        f"N = {stat['N']:,}\n$\\sigma_{{\\rm NMAD}}$ = {stat['sigma_NMAD']:.4f}\noutlier = {stat['outlier_fraction']:.3f}\nbias = {stat['bias']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.76),
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_field_table(field_df: pd.DataFrame, out_path: Path) -> None:
    zfinal = pd.to_numeric(field_df["z_phot"], errors="coerce").to_numpy(np.float32)
    sigma = pd.to_numeric(field_df["z_mc_std"], errors="coerce").to_numpy(np.float32)
    fallback = (0.03 * (1.0 + zfinal)).astype(np.float32)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback).astype(np.float32)

    def fixed_str(values: pd.Series) -> np.ndarray:
        arr = values.astype(str).to_numpy()
        max_len = max(1, max(len(x) for x in arr))
        return np.asarray(arr, dtype=f"U{max_len}")

    tab = Table()
    tab["row_idx"] = pd.to_numeric(field_df["row_idx"], errors="coerce").to_numpy(np.int64)
    tab["id"] = fixed_str(field_df["id"])
    tab["ra"] = pd.to_numeric(field_df["ra"], errors="coerce").to_numpy(np.float32)
    tab["dec"] = pd.to_numeric(field_df["dec"], errors="coerce").to_numpy(np.float32)
    tab["redshift"] = pd.to_numeric(field_df["redshift"], errors="coerce").to_numpy(np.float32)
    tab["z_true"] = pd.to_numeric(field_df["z_true"], errors="coerce").to_numpy(np.float32)
    tab["zfinal"] = zfinal
    tab["zpdf_l68"] = np.clip(zfinal - sigma, 0.0, 3.8).astype(np.float32)
    tab["zpdf_u68"] = np.clip(zfinal + sigma, 0.0, 3.8).astype(np.float32)
    tab["z_i22_cross_lstm"] = zfinal
    tab["z_i22_cross_lstm_sigma"] = sigma
    tab["z_mc_mean"] = pd.to_numeric(field_df["z_mc_mean"], errors="coerce").to_numpy(np.float32)
    tab["zConf"] = pd.to_numeric(field_df["zConf"], errors="coerce").to_numpy(np.float32)
    tab["mag_i"] = pd.to_numeric(field_df["mag_i"], errors="coerce").to_numpy(np.float32)
    tab["magerr_i"] = fluxerr_to_magerr(field_df["f_i"], field_df["e_i"])
    tab["n_valid_bands"] = pd.to_numeric(field_df["n_valid_bands"], errors="coerce").to_numpy(np.int16)
    tab["field_id"] = pd.to_numeric(field_df["field_id"], errors="coerce").to_numpy(np.int16)
    tab["sky_half"] = fixed_str(field_df["sky_half"])
    tab["direction"] = fixed_str(field_df["direction"])
    for band in BANDS:
        tab[f"f_{band}"] = pd.to_numeric(field_df[f"f_{band}"], errors="coerce").to_numpy(np.float32)
        tab[f"e_{band}"] = pd.to_numeric(field_df[f"e_{band}"], errors="coerce").to_numpy(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tab.write(out_path, overwrite=True)


def make_field_plots(summary: pd.DataFrame, merged: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    setup_style()
    count_png = outdir / "i22_cross_lstm_5field_counts_area.png"
    sky_png = outdir / "i22_cross_lstm_5field_sky_distribution.png"

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(summary))
    labels = [f"F{int(v):02d}" for v in summary["field_id"]]
    ax1.bar(x - 0.18, summary["rows"], width=0.36, color="#4C78A8", label="N sources")
    ax1.set_ylabel("N sources")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, summary["area_hull_deg2"], "o-", color="#B2182B", lw=1.4, ms=4, label="Hull area")
    ax2.set_ylabel(r"Hull area (deg$^2$)")
    ax1.set_xlabel("Field")
    fig.tight_layout()
    fig.savefig(count_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B", 4: "#B279A2", 5: "#E45756"}
    for field_id, part in merged.groupby("field_id"):
        sample = part
        if len(sample) > 50000:
            sample = sample.sample(50000, random_state=42)
        ax.scatter(sample["ra"], sample["dec"], s=0.25, alpha=0.35, linewidths=0, color=colors[int(field_id)], label=f"Field {int(field_id):02d}")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.legend(frameon=False, markerscale=6, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(sky_png, bbox_inches="tight")
    plt.close(fig)
    return count_png, sky_png


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.note_path.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    inputs = load_inputs(args.input_dir)
    keep_cols = [
        "row_idx",
        "id",
        "redshift",
        "n_valid_bands",
        "mag_i",
        *[f"{prefix}_{band}" for band in BANDS for prefix in ("f", "e")],
    ]
    merged = pred.merge(inputs[keep_cols], on=["row_idx", "id"], how="left", validate="one_to_one")
    if merged["mag_i"].isna().any():
        raise RuntimeError("Some predictions did not match i<22 input rows.")

    plot_dir = args.output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    hemi_plots = {}
    for half in ["hemisphere_A", "hemisphere_B"]:
        part = merged[(merged["sky_half"] == half) & (merged["sample_role"] == "cross_target")].copy()
        out_png = plot_dir / f"{half}_i22_cross_lstm_photoz_vs_ztrue.png"
        plot_photoz(part, half, out_png)
        hemi_plots[half] = out_png

    rows = []
    for field_id, field_df in merged.groupby("field_id"):
        field_id = int(field_id)
        half = str(field_df["sky_half"].iloc[0])
        out_path = args.output_root / half / f"csst_field_{field_id:02d}_i22_cross_lstm.fits"
        write_field_table(field_df, out_path)
        ra = pd.to_numeric(field_df["ra"], errors="coerce").to_numpy(float)
        dec = pd.to_numeric(field_df["dec"], errors="coerce").to_numpy(float)
        hull, bbox = area_deg2(ra, dec)
        z = pd.to_numeric(field_df["zfinal"], errors="coerce").to_numpy(float) if "zfinal" in field_df else pd.to_numeric(field_df["z_phot"], errors="coerce").to_numpy(float)
        rows.append(
            {
                "field_id": field_id,
                "half": half,
                "rows": len(field_df),
                "ra_min": float(np.nanmin(ra)),
                "ra_max": float(np.nanmax(ra)),
                "dec_min": float(np.nanmin(dec)),
                "dec_max": float(np.nanmax(dec)),
                "area_hull_deg2": hull,
                "area_bbox_cosdec_deg2": bbox,
                "z_phot_median": float(np.nanmedian(z)),
                "mag_i_median": float(np.nanmedian(pd.to_numeric(field_df["mag_i"], errors="coerce"))),
                "output_fits": str(out_path),
            }
        )

    summary = pd.DataFrame(rows).sort_values("field_id").reset_index(drop=True)
    summary_csv = args.output_root / "csst_7band_i22_cross_lstm_5field_summary.csv"
    summary.to_csv(summary_csv, index=False)
    count_png, sky_png = make_field_plots(summary, merged, plot_dir)

    metrics_df = pd.read_csv(args.metrics_csv)
    note = [
        "# CSST 7band i<22 Cross-Hemisphere LSTM Photo-z",
        "",
        "## 目标",
        "",
        "使用 7band 中 `i < 22` 的亮星样本重新做 cross-hemisphere LSTM：",
        "",
        "- hemisphere A 训练，预测 hemisphere B；",
        "- hemisphere B 训练，预测 hemisphere A；",
        "- 不运行 EAZY-py，不计算新的 stellar mass；",
        "- 将 cross-target 预测写成 5 个 field 的最终 blind-search 可用 FITS 星表。",
        "",
        "## 输入与输出",
        "",
        f"- i<22 输入目录：`{args.input_dir}`",
        f"- 预测总表：`{args.pred_csv}`",
        f"- 输出目录：`{args.output_root}`",
        f"- field summary：`{summary_csv}`",
        "",
        "## Photo-z 指标",
        "",
        "| direction | sample_role | N | sigma_NMAD | outlier_fraction | bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics_df.to_dict(orient="records"):
        note.append(
            f"| {row['direction']} | {row['sample_role']} | {int(row['N']):,} | {row['sigma_NMAD']:.6f} | {row['outlier_fraction']:.6f} | {row['bias']:.6f} |"
        )
    note += [
        "",
        "## 半球 photo-z vs z_true",
        "",
        f"![hemisphere A]({hemi_plots['hemisphere_A']})",
        "",
        f"![hemisphere B]({hemi_plots['hemisphere_B']})",
        "",
        "## 五个 field 最终星表",
        "",
        "| field | half | rows | hull_area_deg2 | bbox_area_deg2 | z_phot_median | mag_i_median | FITS |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict(orient="records"):
        note.append(
            f"| {int(row['field_id'])} | {row['half']} | {int(row['rows']):,} | {row['area_hull_deg2']:.3f} | {row['area_bbox_cosdec_deg2']:.3f} | {row['z_phot_median']:.3f} | {row['mag_i_median']:.3f} | `{row['output_fits']}` |"
        )
    note += [
        "",
        "## 五场数据分布图",
        "",
        f"![counts area]({count_png})",
        "",
        f"![sky distribution]({sky_png})",
        "",
        "## 说明",
        "",
        "- `zfinal` 为本轮 `i<22` cross-hemisphere LSTM 的 `z_phot`。",
        "- `zpdf_l68/u68` 由 MC dropout 的 `z_mc_std` 给出；若无有效 scatter，则使用 `0.03*(1+zfinal)` fallback。",
        "- 面积为当前 `i<22` 选中源在各 field 内的 RA-Dec 凸包面积；同时给出 cos(dec) 修正后的 bbox 面积供参考。",
        "- 这轮没有运行 EAZY-py，因此输出 FITS 不包含新的 EAZY stellar mass。",
        "",
    ]
    args.note_path.write_text("\n".join(note), encoding="utf-8")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote note: {args.note_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
