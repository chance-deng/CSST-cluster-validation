#!/usr/bin/env python3
"""Plot a volume-style 3D galaxy density map for Field 05 purity>=90 candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from skimage import measure


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
BASE_DIR = PROJECT_ROOT / "Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage"
PURITY_DIR = BASE_DIR / "field05_i22_purity90_candidates_3d"
FIELD05_FULL7_FITS = (
    PROJECT_ROOT
    / "Codex/result/blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/hemisphere_B/csst_field_05_full7band_cross_lstm.fits"
)
OUT_DIR = BASE_DIR / "field05_i22_purity90_volume_density"
OUT_PNG = OUT_DIR / "field05_i22_all7band_volume_density_candidates.png"
OUT_PDF = OUT_DIR / "field05_i22_all7band_volume_density_candidates.pdf"
OUT_MD = PROJECT_ROOT / "Codex/notebook/field05_i22_purity90_volume_density.md"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
LINK_RPROJ_CMPC = 10.0
LINK_LOS_CMPC = 100.0


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
    # marching_cubes returns coordinates in index order for density axes.
    # Match the preferred view: comoving distance is the long foreground axis,
    # RA is the receding sky axis, and DEC is the vertical sky axis.
    coords = []
    for axis_idx in range(3):
        centers = 0.5 * (edges[axis_idx][:-1] + edges[axis_idx][1:])
        grid_idx = np.arange(len(centers))
        coords.append(np.interp(verts[:, axis_idx], grid_idx, centers))
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


def main() -> int:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(PURITY_DIR / "field05_i22_purity90_candidates.csv")
    matched = cand[cand["matched"]].copy()
    unmatched = cand[~cand["matched"]].copy()
    link_rows = []
    if len(matched) > 1:
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
                link_rows.append(
                    {
                        "id1": int(ids[i]),
                        "id2": int(ids[j]),
                        "rproj_cmpc": float(rproj[offset]),
                        "los_cmpc": float(los[offset]),
                        "i": i,
                        "j": j,
                    }
                )

    tab = Table.read(FIELD05_FULL7_FITS, memmap=True)
    ra = np.asarray(tab["ra"], dtype=float)
    dec = np.asarray(tab["dec"], dtype=float)
    z = np.asarray(tab["zfinal"], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    ra = ra[mask]
    dec = dec[mask]
    dist = COSMO.comoving_distance(z[mask]).value

    # Restrict to the candidate redshift/depth range plus a small margin.  This
    # makes the density cloud describe the volume relevant to the plotted sample.
    dmin = max(float(np.nanmin(dist)), float(cand["comoving_distance_mpc"].min()) - 500.0)
    dmax = min(float(np.nanmax(dist)), float(cand["comoving_distance_mpc"].max()) + 500.0)
    volume_mask = (dist >= dmin) & (dist <= dmax)
    ra = ra[volume_mask]
    dec = dec[volume_mask]
    dist = dist[volume_mask]

    bins = [
        np.linspace(dmin, dmax, 58),
        np.linspace(float(np.nanmin(ra)), float(np.nanmax(ra)), 34),
        np.linspace(float(np.nanmin(dec)), float(np.nanmax(dec)), 34),
    ]
    hist, edges = np.histogramdd(np.column_stack([dist, ra, dec]), bins=bins)
    density = gaussian_filter(hist.astype(float), sigma=(0.2, 0.2, 0.2), mode="constant")
    density = density / np.nanmax(density)

    fig = plt.figure(figsize=(10.8, 7.2), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Low-to-high density shells.  Similar visual language to a translucent 3D
    # density map: purple outskirts, orange/yellow core.
    levels = [0.020, 0.040, 0.075, 0.140]
    colors = ["#4C5AA7", "#6F4AA8", "#C44E9A", "#FEE08B"]
    alphas = [0.022, 0.036, 0.058, 0.120]
    for level, color, alpha in zip(levels, colors, alphas):
        add_isosurface(ax, density, edges, level, color, alpha)

    for link in link_rows:
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

    ax.scatter(
        matched["comoving_distance_mpc"],
        matched["RA"],
        matched["Dec"],
        s=52,
        c="#3C5488",
        edgecolors="black",
        linewidths=0.55,
        marker="o",
        label=f"Matched candidates ({len(matched)})",
        depthshade=False,
        zorder=20,
    )
    ax.scatter(
        unmatched["comoving_distance_mpc"],
        unmatched["RA"],
        unmatched["Dec"],
        s=78,
        c="#E64B35",
        edgecolors="black",
        linewidths=0.65,
        marker="^",
        label=f"Unmatched candidates ({len(unmatched)})",
        depthshade=False,
        zorder=21,
    )

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
        "Field 05 faint 7band galaxy density and high-purity candidates\n"
        rf"All 7band galaxy density; candidates selected with $i<22$, $n_{{\rm members}}\geq67$; links: $R_{{\rm proj}}\leq{LINK_RPROJ_CMPC:g}$ cMpc, $\Delta D\leq{LINK_LOS_CMPC:g}$ cMpc",
        fontsize=13.2,
        pad=16,
    )
    ax.tick_params(axis="both", which="major", pad=3, labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.04), ncol=2, frameon=False, fontsize=10)
    fig.text(0.875, 0.49, "DEC [J2000] (deg)", rotation=90, va="center", ha="center", fontsize=13)
    fig.subplots_adjust(left=0.015, right=0.965, bottom=0.08, top=0.86)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.28)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)

    OUT_MD.write_text(
        "\n".join(
            [
                "# Field 05 i<22 High-purity Candidate Volume Density Map",
                "",
                "- Background density: all full 7band Field 05 galaxies, without an i-band cut.",
                "- Density construction: 3D histogram in `comoving distance, RA, DEC`, Gaussian smoothed with `sigma=0.2` and rendered as translucent isosurfaces.",
                "- Candidate selection: `i<22`, `n_members >= 67`, the first threshold with purity proxy >= 90%.",
                f"- Red links connect matched candidates only when projected separation is `<= {LINK_RPROJ_CMPC:g} cMpc` and line-of-sight comoving-distance separation is `<= {LINK_LOS_CMPC:g} cMpc`.",
                f"- Linked matched candidate pairs: `{len(link_rows)}`.",
                f"- Candidates: `{len(cand)}`; matched: `{len(matched)}`; unmatched: `{len(unmatched)}`.",
                f"- Figure: `{OUT_PNG}`",
                "",
                f"![field05 volume density]({OUT_PNG})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
