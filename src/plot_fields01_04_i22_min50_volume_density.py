#!/usr/bin/env python3
"""Plot Field 01-04 i<22 candidate volume-density maps with at least 50 candidates."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from skimage import measure


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
BASE_DIR = PROJECT_ROOT / "Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage"
SCAN_CSV = BASE_DIR / "extended_i22_nmembers_scan_to_10/i22_nmembers_extended_scan_all_fields.csv"
FIELD_SUMMARY = PROJECT_ROOT / "Codex/result/blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/full7band_cross_lstm_5field_summary.csv"
OUT_DIR = BASE_DIR / "fields01_04_i22_min50_volume_density"
OUT_MD = PROJECT_ROOT / "Codex/notebook/fields01_04_i22_min50_volume_density.md"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
LINK_RPROJ_CMPC = 10.0
LINK_LOS_CMPC = 100.0
SMOOTH_SIGMA = 0.2


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.linewidth": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
        }
    )


def add_isosurface(ax, density: np.ndarray, edges: list[np.ndarray], level: float, color, alpha: float) -> int:
    if not (np.nanmin(density) < level < np.nanmax(density)):
        return 0
    verts, faces, _, _ = measure.marching_cubes(density, level=level)
    coords = []
    for axis_idx in range(3):
        centers = 0.5 * (edges[axis_idx][:-1] + edges[axis_idx][1:])
        coords.append(np.interp(verts[:, axis_idx], np.arange(len(centers)), centers))
    xyz = np.column_stack(coords)
    mesh = ax.plot_trisurf(
        xyz[:, 0],
        xyz[:, 1],
        faces,
        xyz[:, 2],
        color=color,
        alpha=alpha,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        zorder=1,
    )
    mesh.set_edgecolor((1, 1, 1, 0))
    return len(faces)


def choose_threshold(scan: pd.DataFrame, field_id: int) -> pd.Series:
    sub = scan[(scan["field_id"] == field_id) & (np.isclose(scan["i_limit"], 22.0)) & (scan["candidates_kept"] >= 50)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"Field {field_id:02d} has no i<22 threshold with >=50 candidates.")
    return sub.sort_values("n_members_threshold").iloc[-1]


def build_links(matched: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if len(matched) <= 1:
        return rows
    coords = SkyCoord(ra=matched["RA"].to_numpy(float) * u.deg, dec=matched["Dec"].to_numpy(float) * u.deg)
    dcom = matched["comoving_distance_mpc"].to_numpy(float)
    ids = matched["ID"].to_numpy(int)
    for i in range(len(matched) - 1):
        sep = coords[i].separation(coords[i + 1 :]).radian
        mean_d = 0.5 * (dcom[i] + dcom[i + 1 :])
        rproj = sep * mean_d
        los = np.abs(dcom[i] - dcom[i + 1 :])
        ok = (rproj <= LINK_RPROJ_CMPC) & (los <= LINK_LOS_CMPC)
        for offset in np.where(ok)[0]:
            j = i + 1 + int(offset)
            rows.append({"id1": int(ids[i]), "id2": int(ids[j]), "rproj_cmpc": float(rproj[offset]), "los_cmpc": float(los[offset]), "i": i, "j": j})
    return rows


def plot_field(field_id: int, threshold_row: pd.Series, field_fits: Path) -> dict:
    out_png = OUT_DIR / f"field{field_id:02d}_i22_min50_volume_density_candidates.png"
    out_pdf = OUT_DIR / f"field{field_id:02d}_i22_min50_volume_density_candidates.pdf"
    out_candidates = OUT_DIR / f"field{field_id:02d}_i22_min50_candidates.csv"
    threshold = int(threshold_row["n_members_threshold"])

    cand = pd.read_csv(threshold_row["candidate_csv"])
    pairs = pd.read_csv(threshold_row["match_csv"])
    matched_ids = set(pairs["candidate_id"].astype(int).tolist())
    kept = cand[cand["n_members"] >= threshold].copy().reset_index(drop=True)
    kept["matched"] = kept["ID"].astype(int).isin(matched_ids)
    kept["match_status"] = np.where(kept["matched"], "matched", "unmatched")
    kept["comoving_distance_mpc"] = COSMO.comoving_distance(kept["z_peak"].to_numpy(float)).value
    kept.to_csv(out_candidates, index=False)

    matched = kept[kept["matched"]].copy().reset_index(drop=True)
    unmatched = kept[~kept["matched"]].copy().reset_index(drop=True)
    links = build_links(matched)

    tab = Table.read(field_fits, memmap=True)
    ra = np.asarray(tab["ra"], dtype=float)
    dec = np.asarray(tab["dec"], dtype=float)
    z = np.asarray(tab["zfinal"], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    ra = ra[mask]
    dec = dec[mask]
    dist = COSMO.comoving_distance(z[mask]).value
    dmin = max(float(np.nanmin(dist)), float(kept["comoving_distance_mpc"].min()) - 500.0)
    dmax = min(float(np.nanmax(dist)), float(kept["comoving_distance_mpc"].max()) + 500.0)
    vm = (dist >= dmin) & (dist <= dmax)
    ra = ra[vm]
    dec = dec[vm]
    dist = dist[vm]

    bins = [
        np.linspace(dmin, dmax, 58),
        np.linspace(float(np.nanmin(ra)), float(np.nanmax(ra)), 34),
        np.linspace(float(np.nanmin(dec)), float(np.nanmax(dec)), 34),
    ]
    hist, edges = np.histogramdd(np.column_stack([dist, ra, dec]), bins=bins)
    density = gaussian_filter(hist.astype(float), sigma=(SMOOTH_SIGMA, SMOOTH_SIGMA, SMOOTH_SIGMA), mode="constant")
    density = density / np.nanmax(density)

    fig = plt.figure(figsize=(10.8, 7.2), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    levels = [0.020, 0.040, 0.075, 0.140]
    colors = ["#4C5AA7", "#6F4AA8", "#C44E9A", "#FEE08B"]
    alphas = [0.022, 0.036, 0.058, 0.120]
    for level, color, alpha in zip(levels, colors, alphas):
        add_isosurface(ax, density, edges, level, color, alpha)

    for link in links:
        p1 = matched.iloc[int(link["i"])]
        p2 = matched.iloc[int(link["j"])]
        ax.plot(
            [p1["comoving_distance_mpc"], p2["comoving_distance_mpc"]],
            [p1["RA"], p2["RA"]],
            [p1["Dec"], p2["Dec"]],
            color="#E64B35",
            lw=0.85,
            alpha=0.58,
            zorder=18,
        )

    ax.scatter(matched["comoving_distance_mpc"], matched["RA"], matched["Dec"], s=52, c="#3C5488", edgecolors="black", linewidths=0.55, marker="o", label=f"Matched candidates ({len(matched)})", depthshade=False, zorder=20)
    ax.scatter(unmatched["comoving_distance_mpc"], unmatched["RA"], unmatched["Dec"], s=78, c="#E64B35", edgecolors="black", linewidths=0.65, marker="^", label=f"Unmatched candidates ({len(unmatched)})", depthshade=False, zorder=21)

    ax.set_xlabel("Comoving Distance (Mpc)", labelpad=15)
    ax.set_ylabel("RA [J2000] (deg)", labelpad=14)
    ax.set_zlabel("")
    ax.zaxis.set_rotate_label(False)
    ax.set_box_aspect((3, 1, 1))
    ax.view_init(elev=23, azim=-48)
    ax.invert_xaxis()
    ax.invert_yaxis()
    pane = (0.88, 0.93, 0.98, 0.82)
    grid = (1.0, 1.0, 1.0, 0.85)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor((0.78, 0.82, 0.88, 0.85))
        axis._axinfo["grid"]["color"] = grid
        axis._axinfo["grid"]["linewidth"] = 0.75
    ax.set_title(
        rf"Field {field_id:02d}: all 7band galaxy density; $i<22$, $n_{{\rm members}}\geq{threshold}$",
        fontsize=13.2,
        pad=16,
    )
    ax.tick_params(axis="both", which="major", pad=3, labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.04), ncol=2, frameon=False, fontsize=10)
    fig.text(0.875, 0.49, "DEC [J2000] (deg)", rotation=90, va="center", ha="center", fontsize=13)
    fig.subplots_adjust(left=0.015, right=0.965, bottom=0.08, top=0.86)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.28)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)

    return {
        "field_id": field_id,
        "n_members_threshold": threshold,
        "candidates_kept": int(len(kept)),
        "matched_candidates": int(len(matched)),
        "unmatched_candidates": int(len(unmatched)),
        "matched_true_clusters": int(threshold_row["matched_true_clusters"]),
        "match_rate": float(threshold_row["match_rate"]),
        "purity_proxy": float(threshold_row["purity_proxy"]),
        "linked_pairs": int(len(links)),
        "figure_png": str(out_png),
        "candidate_csv": str(out_candidates),
    }


def main() -> int:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan = pd.read_csv(SCAN_CSV)
    field_summary = pd.read_csv(FIELD_SUMMARY).set_index("field_id")
    rows = []
    for field_id in [1, 2, 3, 4]:
        selected = choose_threshold(scan, field_id)
        print(
            f"Field {field_id:02d}: n_members>={int(selected['n_members_threshold'])}, "
            f"candidates={int(selected['candidates_kept'])}, purity={selected['purity_proxy']:.2%}",
            flush=True,
        )
        rows.append(plot_field(field_id, selected, Path(field_summary.loc[field_id, "output_fits"])))

    summary = pd.DataFrame(rows)
    summary_csv = OUT_DIR / "fields01_04_i22_min50_volume_density_summary.csv"
    summary.to_csv(summary_csv, index=False)
    lines = [
        "# Fields 01-04 i<22 Minimum-50 Candidate Volume Density Maps",
        "",
        "- Candidate selection: highest integer `n_members` threshold that keeps at least 50 candidates.",
        "- Density background: all full 7band galaxies in the corresponding field, no i-band cut.",
        f"- Gaussian smoothing: `sigma={SMOOTH_SIGMA}`.",
        f"- Red links connect matched candidates only when `R_proj <= {LINK_RPROJ_CMPC:g} cMpc` and `Delta D <= {LINK_LOS_CMPC:g} cMpc`.",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "| field | n_members >= | candidates | matched cand | unmatched cand | matched true clusters | match rate | purity proxy | linked pairs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | {int(row['n_members_threshold'])} | {int(row['candidates_kept'])} | "
            f"{int(row['matched_candidates'])} | {int(row['unmatched_candidates'])} | {int(row['matched_true_clusters'])} | "
            f"{row['match_rate']:.2%} | {row['purity_proxy']:.2%} | {int(row['linked_pairs'])} |"
        )
    lines.append("")
    for _, row in summary.iterrows():
        lines.extend([f"## Field {int(row['field_id']):02d}", "", f"![field{int(row['field_id']):02d}]({row['figure_png']})", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
