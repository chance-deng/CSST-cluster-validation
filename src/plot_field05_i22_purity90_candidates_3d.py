#!/usr/bin/env python3
"""Plot Field 05 i<22 candidates at the first >=90% purity n_members threshold."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
SCAN_DIR = (
    PROJECT_ROOT
    / "Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage/extended_i22_nmembers_scan_to_10"
)
TIGHT_ROOT = PROJECT_ROOT / "Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage"
OUT_DIR = TIGHT_ROOT / "field05_i22_purity90_candidates_3d"
OUT_PNG = OUT_DIR / "field05_i22_purity90_candidates_3d.png"
OUT_PDF = OUT_DIR / "field05_i22_purity90_candidates_3d.pdf"
OUT_DISTANCE_PNG = OUT_DIR / "field05_i22_purity90_candidates_3d_comoving_distance.png"
OUT_DISTANCE_PDF = OUT_DIR / "field05_i22_purity90_candidates_3d_comoving_distance.pdf"
OUT_CSV = OUT_DIR / "field05_i22_purity90_candidates.csv"
OUT_MD = PROJECT_ROOT / "Codex/notebook/field05_i22_purity90_candidates_3d.md"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
FIELD05_FULL7_FITS = (
    PROJECT_ROOT
    / "Codex/result/blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/hemisphere_B/csst_field_05_full7band_cross_lstm.fits"
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
        }
    )


def main() -> int:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scan = pd.read_csv(SCAN_DIR / "i22_nmembers_extended_scan_all_fields.csv")
    field_scan = scan[(scan["field_id"] == 5) & (scan["purity_proxy"] >= 0.90)].copy()
    if len(field_scan) == 0:
        raise RuntimeError("No field05 i<22 threshold reaches purity_proxy >= 90%.")
    # Smallest threshold that reaches purity >= 90 keeps the largest candidate sample.
    chosen = field_scan.sort_values("n_members_threshold").iloc[0]
    threshold = int(chosen["n_members_threshold"])

    candidate_csv = Path(chosen["candidate_csv"])
    match_csv = Path(chosen["match_csv"])
    cand = pd.read_csv(candidate_csv)
    pairs = pd.read_csv(match_csv)
    matched_ids = set(pairs["candidate_id"].astype(int).tolist())

    kept = cand[cand["n_members"] >= threshold].copy().reset_index(drop=True)
    kept["matched"] = kept["ID"].astype(int).isin(matched_ids)
    kept["match_status"] = np.where(kept["matched"], "matched", "unmatched")
    kept["comoving_distance_mpc"] = COSMO.comoving_distance(kept["z_peak"].to_numpy(float)).value
    kept.to_csv(OUT_CSV, index=False)

    matched = kept[kept["matched"]].copy()
    unmatched = kept[~kept["matched"]].copy()

    field05 = Table.read(FIELD05_FULL7_FITS, memmap=True)
    bg_ra = np.asarray(field05["ra"], dtype=float)
    bg_dec = np.asarray(field05["dec"], dtype=float)
    bg_z = np.asarray(field05["zfinal"], dtype=float)
    bg_mag_i = np.asarray(field05["mag_i"], dtype=float)
    bg_mask = np.isfinite(bg_ra) & np.isfinite(bg_dec) & np.isfinite(bg_z) & np.isfinite(bg_mag_i) & (bg_mag_i > 22.0)
    bg_ra = bg_ra[bg_mask]
    bg_dec = bg_dec[bg_mask]
    bg_dist = COSMO.comoving_distance(bg_z[bg_mask]).value

    fig = plt.figure(figsize=(8.2, 6.8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    if len(matched):
        ax.scatter(
            matched["RA"],
            matched["Dec"],
            matched["z_peak"],
            s=38,
            c="#3C5488",
            alpha=0.78,
            edgecolors="black",
            linewidths=0.35,
            label=f"Matched ({len(matched)})",
            depthshade=True,
        )
    if len(unmatched):
        ax.scatter(
            unmatched["RA"],
            unmatched["Dec"],
            unmatched["z_peak"],
            s=58,
            c="#E64B35",
            alpha=0.95,
            marker="^",
            edgecolors="black",
            linewidths=0.55,
            label=f"Unmatched ({len(unmatched)})",
            depthshade=True,
        )

    ax.set_xlabel("RA (deg)", labelpad=8)
    ax.set_ylabel("Dec (deg)", labelpad=8)
    ax.set_zlabel(r"$z_{\rm peak}$", labelpad=8)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(
        f"Field 05 i<22.0 candidates with purity >=90%\n"
        f"n_members >= {threshold}, candidates={len(kept)}, purity={chosen['purity_proxy']:.2%}, "
        f"match rate={chosen['match_rate']:.2%}",
        fontsize=13,
        pad=14,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    # Replot with the redshift axis converted to comoving distance.  This view
    # follows the paper-style 3D layout: distance is the long horizontal axis,
    # RA is the receding axis, and Dec is vertical.
    fig = plt.figure(figsize=(11.2, 7.4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    # Light 3D density proxy for faint 7band galaxies.  Plot as low-alpha
    # density-cell centers before candidates so it provides context without
    # covering the candidate markers.
    bins = [
        np.linspace(np.nanmin(bg_dist), np.nanmax(bg_dist), 34),
        np.linspace(np.nanmin(bg_dec), np.nanmax(bg_dec), 14),
        np.linspace(np.nanmin(bg_ra), np.nanmax(bg_ra), 14),
    ]
    hist, edges = np.histogramdd(
        np.column_stack([bg_dist, bg_dec, bg_ra]),
        bins=bins,
    )
    nz = hist > 0
    if np.any(nz):
        x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
        y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
        z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])
        ix, iy, iz = np.where(nz)
        dens = hist[nz]
        size = 3.0 + 22.0 * np.sqrt(dens / np.nanmax(dens))
        ax.scatter(
            x_centers[ix],
            y_centers[iy],
            z_centers[iz],
            s=size,
            c="#8FB6D8",
            alpha=0.105,
            linewidths=0,
            label=r"7band galaxies ($i>22$) density",
            depthshade=False,
            zorder=1,
        )
    if len(matched):
        ax.scatter(
            matched["comoving_distance_mpc"],
            matched["Dec"],
            matched["RA"],
            s=38,
            c="#3C5488",
            alpha=0.78,
            edgecolors="black",
            linewidths=0.35,
            label=f"Matched ({len(matched)})",
            depthshade=True,
        )
    if len(unmatched):
        ax.scatter(
            unmatched["comoving_distance_mpc"],
            unmatched["Dec"],
            unmatched["RA"],
            s=58,
            c="#E64B35",
            alpha=0.95,
            marker="^",
            edgecolors="black",
            linewidths=0.55,
            label=f"Unmatched ({len(unmatched)})",
            depthshade=True,
        )

    ax.set_xlabel("Comoving Distance (Mpc)", labelpad=14)
    ax.set_ylabel("RA [J2000] (deg)", labelpad=14)
    ax.set_zlabel("DEC [J2000] (deg)", labelpad=24)
    ax.zaxis.set_rotate_label(False)
    ax.set_box_aspect((3, 1, 1))
    ax.view_init(elev=19, azim=-47)
    ax.invert_xaxis()
    ax.invert_yaxis()

    pane_color = (0.86, 0.93, 0.98, 0.72)
    grid_color = (0.62, 0.69, 0.75, 0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_edgecolor((0.50, 0.56, 0.62, 0.65))
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.65
        axis._axinfo["axisline"]["color"] = (0.20, 0.20, 0.20, 1.0)

    ax.set_title(
        f"Field 05 i<22.0 candidates with purity >=90%\n"
        f"n_members >= {threshold}, candidates={len(kept)}, purity={chosen['purity_proxy']:.2%}, "
        "redshift axis shown as comoving-depth",
        fontsize=13,
        pad=14,
    )
    ax.tick_params(axis="both", which="major", pad=3, labelsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8.5, markerscale=1.6)
    ax.grid(True, alpha=0.25)
    fig.text(0.965, 0.49, "DEC [J2000] (deg)", rotation=90, va="center", ha="center", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.09, top=0.88)
    fig.savefig(OUT_DISTANCE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.42)
    fig.savefig(OUT_DISTANCE_PDF, bbox_inches="tight", pad_inches=0.42)
    plt.close(fig)

    lines = [
        "# Field 05 i<22.0 Purity >=90% Candidate 3D Distribution",
        "",
        f"- Candidate table: `{candidate_csv}`",
        f"- Match table: `{match_csv}`",
        f"- Selected threshold: `n_members >= {threshold}`",
        f"- Candidates kept: `{len(kept)}`",
        f"- Matched candidates: `{len(matched)}`",
        f"- Unmatched candidates: `{len(unmatched)}`",
        f"- Purity proxy: `{chosen['purity_proxy']:.2%}`",
        f"- Match rate: `{chosen['match_rate']:.2%}`",
        f"- Output candidate CSV: `{OUT_CSV}`",
        "",
        f"![field05 purity90 3D]({OUT_PNG})",
        "",
        "## Comoving-distance depth-axis version",
        "",
        "- The redshift axis is converted to comoving distance using `FlatLambdaCDM(H0=70, Om0=0.3)`.",
        "- Axis aspect is set to `RA : comoving distance : Dec = 1 : 3 : 1`, i.e. the redshift/depth axis is three times longer.",
        "- A faint blue density layer shows full 7band Field 05 galaxies with `mag_i > 22`, binned in 3D and drawn behind the candidates.",
        "",
        f"![field05 purity90 3D comoving]({OUT_DISTANCE_PNG})",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"threshold={threshold}")
    print(f"candidates={len(kept)} matched={len(matched)} unmatched={len(unmatched)}")
    print(OUT_PNG)
    print(OUT_DISTANCE_PNG)
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
