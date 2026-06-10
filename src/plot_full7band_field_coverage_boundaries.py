#!/usr/bin/env python3
"""Plot full-7band galaxy coverage and tight/convex boundaries for each field."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
CODE_DIR = PROJECT_ROOT / "Codex/code"
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"
FIELD_SUMMARY = RESULT_ROOT / "blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/full7band_cross_lstm_5field_summary.csv"
OUT_ROOT = RESULT_ROOT / "full7band_field_coverage_boundaries"
OUT_MD = NOTEBOOK_ROOT / "full7band_field_coverage_boundaries.md"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.linewidth": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
        }
    )


def polygon_area_xy(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def plot_one(field_id: int, field_path: Path, out_png: Path) -> dict:
    sys.path.insert(0, str(CODE_DIR))
    import check_full7band_irregular_field_coverage as coverage
    import reanalyze_field05_v11_crossmatch_r1p5_and_thresholds as base

    df = coverage.load_field(field_path)
    center_ra = float(np.median(df["ra"]))
    center_dec = float(np.median(df["dec"]))
    x = (df["ra"].to_numpy(float) - center_ra) * np.cos(np.radians(center_dec))
    y = df["dec"].to_numpy(float) - center_dec

    mask, xmin, ymin, nx, ny = coverage.make_grid_mask(x, y)
    polygon, _, _ = base.build_footprint_polygon(df["ra"].to_numpy(float), df["dec"].to_numpy(float), coverage.CONVEX_BUFFER_DEG)

    xedges = xmin + np.arange(nx + 1) * coverage.CELL_SIZE_DEG
    yedges = ymin + np.arange(ny + 1) * coverage.CELL_SIZE_DEG
    counts, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    counts = counts.astype(float)
    counts[counts <= 0] = np.nan
    tight_area = float(mask.sum() * coverage.CELL_SIZE_DEG * coverage.CELL_SIZE_DEG)
    convex_area = polygon_area_xy(polygon)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    im = ax.imshow(
        np.log10(counts + 1.0),
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    xx = xmin + (np.arange(mask.shape[1]) + 0.5) * coverage.CELL_SIZE_DEG
    yy = ymin + (np.arange(mask.shape[0]) + 0.5) * coverage.CELL_SIZE_DEG
    ax.contour(xx, yy, mask.astype(float), levels=[0.5], colors=["#00A087"], linewidths=2.1)
    ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]], color="#3C5488", lw=1.6, ls="--")

    ax.text(
        0.025,
        0.975,
        f"Field {field_id:02d}\nN={len(df):,}\nTight area={tight_area:.2f} deg$^2$\nConvex area={convex_area:.2f} deg$^2$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.28", "alpha": 0.86},
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label(r"$\log_{10}(N_{\rm gal}+1)$ per cell")
    ax.set_title(f"Full 7band Field {field_id:02d} Coverage", fontsize=16)
    ax.set_xlabel(r"$\Delta \mathrm{R.A.}\cos(\mathrm{Dec})$ (deg)")
    ax.set_ylabel(r"$\Delta \mathrm{Dec}$ (deg)")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "field_id": field_id,
        "galaxies": len(df),
        "tight_area_deg2": tight_area,
        "convex_area_deg2": convex_area,
        "area_ratio": tight_area / convex_area if convex_area else np.nan,
        "plot_png": str(out_png),
    }


def main() -> int:
    setup_style()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    field_summary = pd.read_csv(FIELD_SUMMARY)
    rows = []
    for _, row in field_summary.iterrows():
        field_id = int(row["field_id"])
        out_png = OUT_ROOT / f"field{field_id:02d}_full7band_coverage_boundary.png"
        print(f"Plotting field {field_id:02d}: {out_png}")
        rows.append(plot_one(field_id, Path(row["output_fits"]), out_png))
    summary = pd.DataFrame(rows)
    summary_csv = OUT_ROOT / "full7band_field_coverage_boundary_summary.csv"
    summary.to_csv(summary_csv, index=False)

    lines = [
        "# Full 7band Field Coverage Boundaries",
        "",
        "- 背景颜色：full 7band 星系数密度，单位为每个 coverage grid cell 的 `log10(N_gal+1)`。",
        "- 绿色实线：tight coverage 边界，用于重新定义 covered true clusters。",
        "- 蓝色虚线：原 convex hull + buffer 边界，仅用于对比。",
        f"- 统计表：`{summary_csv}`",
        "",
        "| field | galaxies | tight area deg2 | convex area deg2 | tight/convex area ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | {int(row['galaxies']):,} | {row['tight_area_deg2']:.3f} | "
            f"{row['convex_area_deg2']:.3f} | {row['area_ratio']:.2f} |"
        )
    lines.append("")
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"## Field {int(row['field_id']):02d}",
                "",
                f"![field{int(row['field_id']):02d} coverage boundary]({row['plot_png']})",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
