#!/usr/bin/env python3
"""Validate blind-search candidates against covered true clusters for one field."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    }
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
HUBBLE_H = 0.7
PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
CLUSTER_FITS = PROJECT_ROOT / "SMG_trace_ov_code/data/galaxy_clusters.fits"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-id", required=True, help="Field label such as 01 or 02")
    parser.add_argument("--field-fits", required=True, help="Field galaxy FITS path")
    parser.add_argument("--candidate-csv", required=True, help="Blind-search candidate CSV path")
    parser.add_argument("--result-root", required=True, help="Output result directory")
    parser.add_argument("--summary-md", required=True, help="Output markdown summary path")
    parser.add_argument("--buffer-deg", type=float, default=0.03, help="Outward footprint buffer in degrees")
    parser.add_argument("--match-radius-mpc", type=float, default=1.0, help="Projected match radius in pMpc/h")
    parser.add_argument("--dz-factor", type=float, default=0.05, help="Redshift match window factor")
    return parser.parse_args()


def ensure_dirs(result_root: Path, summary_md: Path) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)


def build_footprint_polygon(ra: np.ndarray, dec: np.ndarray, buffer_deg: float) -> tuple[np.ndarray, float, float]:
    center_ra = float(np.median(ra))
    center_dec = float(np.median(dec))
    x = (ra - center_ra) * np.cos(np.radians(center_dec))
    y = dec - center_dec
    points = np.column_stack([x, y])
    if len(points) < 3:
        return points, center_ra, center_dec
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    radii = np.hypot(hull_points[:, 0], hull_points[:, 1])
    angles = np.arctan2(hull_points[:, 1], hull_points[:, 0])
    radii = radii + buffer_deg
    buffered = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return buffered, center_ra, center_dec


def points_in_polygon(xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    path = MplPath(polygon_xy, closed=True)
    return path.contains_points(xy, radius=1e-10)


def load_field_catalog(field_fits: Path, buffer_deg: float) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    tab = Table.read(field_fits)
    df = tab.to_pandas()
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["zfinal"])].copy()
    polygon, center_ra, center_dec = build_footprint_polygon(df["ra"].to_numpy(), df["dec"].to_numpy(), buffer_deg)
    return df, polygon, center_ra, center_dec


def load_true_clusters(
    polygon_xy: np.ndarray,
    field_df: pd.DataFrame,
    center_ra: float,
    center_dec: float,
) -> pd.DataFrame:
    tab = Table.read(CLUSTER_FITS)
    df = tab.to_pandas()
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["redshift"])].copy()
    cluster_xy = np.column_stack(
        [
            (df["ra"].to_numpy() - center_ra) * np.cos(np.radians(center_dec)),
            df["dec"].to_numpy() - center_dec,
        ]
    )
    inside_mask = points_in_polygon(cluster_xy, polygon_xy)
    covered = df[inside_mask].copy().reset_index(drop=True)
    covered["cluster_index"] = np.arange(len(covered), dtype=int)
    return covered


def load_candidates(candidate_csv: Path) -> pd.DataFrame:
    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")
    df = pd.read_csv(candidate_csv)
    df = df[df["z_peak"] <= 3.7].copy().reset_index(drop=True)
    df["cand_index"] = np.arange(len(df), dtype=int)
    return df


def cross_match(
    df_clusters: pd.DataFrame,
    df_cand: pd.DataFrame,
    match_radius_mpc: float,
    dz_factor: float,
) -> tuple[pd.DataFrame, dict]:
    matched_rows = []
    matched_cluster_indices: set[int] = set()
    matched_cand_indices: set[int] = set()
    distances_list = []
    abs_dz_list = []
    pair_count = 0

    coords_groups = SkyCoord(ra=df_clusters["ra"].to_numpy() * u.deg, dec=df_clusters["dec"].to_numpy() * u.deg)
    coords_cands = SkyCoord(ra=df_cand["RA"].to_numpy() * u.deg, dec=df_cand["Dec"].to_numpy() * u.deg)

    for i, group in df_clusters.iterrows():
        gz = float(group["redshift"])
        dz_limit = dz_factor * (1 + gz)
        mask_z = np.abs(df_cand["z_peak"].to_numpy() - gz) <= dz_limit
        potential_indices = np.where(mask_z)[0]
        if len(potential_indices) == 0:
            continue

        da = COSMO.angular_diameter_distance(gz).value
        seps = coords_groups[i].separation(coords_cands[potential_indices])
        dist_mpc_h = seps.radian * da * HUBBLE_H
        mask_r = dist_mpc_h <= match_radius_mpc
        if not np.any(mask_r):
            continue

        local_valid_idx = potential_indices[mask_r]
        local_dists = dist_mpc_h[mask_r]
        matched_cluster_indices.add(int(group["cluster_index"]))
        for cand_idx, dist_val in zip(local_valid_idx, local_dists):
            closest_cand = df_cand.iloc[int(cand_idx)]
            matched_cand_indices.add(int(closest_cand["cand_index"]))
            abs_dz = float(np.abs(float(closest_cand["z_peak"]) - gz))
            distances_list.append(float(dist_val))
            abs_dz_list.append(abs_dz)
            pair_count += 1

            dra = (float(closest_cand["RA"]) - float(group["ra"])) * np.cos(np.radians(float(group["dec"])))
            ddec = float(closest_cand["Dec"]) - float(group["dec"])
            dx = float(np.radians(dra) * da * HUBBLE_H)
            dy = float(np.radians(ddec) * da * HUBBLE_H)

            matched_rows.append(
                {
                    "cluster_index": int(group["cluster_index"]),
                    "bcg_id": group["bcg_id"],
                    "cluster_ra": float(group["ra"]),
                    "cluster_dec": float(group["dec"]),
                    "cluster_z": gz,
                    "candidate_id": int(closest_cand["ID"]),
                    "candidate_index": int(closest_cand["cand_index"]),
                    "candidate_ra": float(closest_cand["RA"]),
                    "candidate_dec": float(closest_cand["Dec"]),
                    "candidate_z_peak": float(closest_cand["z_peak"]),
                    "distance_mpc_h": float(dist_val),
                    "abs_dz": abs_dz,
                    "dx_mpc_h": dx,
                    "dy_mpc_h": dy,
                }
            )

    stats = {
        "total_groups": int(len(df_clusters)),
        "matched_groups": int(len(matched_cluster_indices)),
        "total_candidates": int(len(df_cand)),
        "matched_candidates": int(len(matched_cand_indices)),
        "matched_pairs": int(pair_count),
        "mean_distance_mpc_h": float(np.mean(distances_list)) if distances_list else np.nan,
        "median_distance_mpc_h": float(np.median(distances_list)) if distances_list else np.nan,
        "mean_abs_dz": float(np.mean(abs_dz_list)) if abs_dz_list else np.nan,
        "median_abs_dz": float(np.median(abs_dz_list)) if abs_dz_list else np.nan,
    }
    return pd.DataFrame(matched_rows), stats


def plot_sky(
    field_id: str,
    field_df: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_cand: pd.DataFrame,
    polygon_xy: np.ndarray,
    plot_path: Path,
) -> None:
    center_ra = float(np.median(field_df["ra"]))
    center_dec = float(np.median(field_df["dec"]))
    hull_ra = center_ra + polygon_xy[:, 0] / np.cos(np.radians(center_dec))
    hull_dec = center_dec + polygon_xy[:, 1]

    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=300)
    ax.scatter(field_df["ra"], field_df["dec"], s=2.5, c="black", alpha=0.45, linewidths=0, label="Raw Galaxies")
    ax.scatter(
        df_clusters["ra"],
        df_clusters["dec"],
        s=165,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="True Clusters",
        zorder=3,
    )
    ax.scatter(
        df_cand["RA"],
        df_cand["Dec"],
        s=1.8,
        c="#00FF33",
        alpha=0.45,
        linewidths=0,
        label="PPM Candidates",
        zorder=2,
    )
    ax.plot(
        np.r_[hull_ra, hull_ra[0]],
        np.r_[hull_dec, hull_dec[0]],
        color="black",
        lw=1.8,
        label=f"Field {field_id} Footprint",
    )
    ax.set_title(f"Field {field_id} Raw Galaxies", fontsize=19)
    ax.set_xlabel("RA (deg)", fontsize=16)
    ax.set_ylabel("Dec (deg)", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_offsets(match_df: pd.DataFrame, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=300)
    circle1 = plt.Circle((0, 0), 1.0, color="#D62728", fill=False, lw=1.5, label="1.0 Mpc/$h$")
    circle2 = plt.Circle((0, 0), 0.5, color="#D62728", fill=False, ls="--", lw=1.5, label="0.5 Mpc/$h$")
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    sc = ax.scatter(
        match_df["dx_mpc_h"],
        match_df["dy_mpc_h"],
        c=match_df["abs_dz"],
        cmap="viridis_r",
        s=22,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    ax.axhline(0, color="black", lw=1.0, ls=":", alpha=0.5)
    ax.axvline(0, color="black", lw=1.0, ls=":", alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta \mathrm{R.A. \ (Mpc/h)}$")
    ax.set_ylabel(r"$\Delta \mathrm{Dec. \ (Mpc/h)}$")
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$|\Delta z|$")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    field_id: str,
    field_fits: Path,
    candidate_csv: Path,
    result_root: Path,
    summary_csv: Path,
    match_csv: Path,
    plot_sky_path: Path,
    plot_offset_path: Path,
    summary_md: Path,
    stats: dict,
    df_clusters: pd.DataFrame,
    df_cand: pd.DataFrame,
    match_df: pd.DataFrame,
    match_radius_mpc: float,
    dz_factor: float,
) -> None:
    match_rate = stats["matched_groups"] / stats["total_groups"] if stats["total_groups"] else np.nan
    cand_rate = stats["matched_candidates"] / stats["total_candidates"] if stats["total_candidates"] else np.nan
    lines = [
        f"# Field {field_id} Cross-match Validation",
        "",
        "## Inputs",
        "",
        f"- Field galaxy catalog: `{field_fits}`",
        f"- Blind-search candidate table: `{candidate_csv}`",
        f"- Cluster reference catalog: `{CLUSTER_FITS}`",
        "",
        "## Covered True Clusters",
        "",
        f"- True clusters are defined as entries from `galaxy_clusters.fits` whose `(ra, dec)` fall inside the field {field_id} sky footprint polygon derived from the field galaxy distribution.",
        f"- Covered true clusters: `{len(df_clusters)}`",
        f"- Blind-search candidates with `z_peak <= 3.7`: `{len(df_cand)}`",
        "",
        "## Matching Rule",
        "",
        f"- Redshift cut: `|z_peak - z_cluster| <= {dz_factor:.3f} * (1 + z_cluster)`",
        f"- Projected-distance cut: `d_proj <= {match_radius_mpc:.1f} pMpc/h`",
        "- If multiple candidates satisfy the cuts for one cluster, keep all of them.",
        "- `Matched true clusters` and `matched candidates` count unique sources with at least one valid pair.",
        f"- `Matched pairs` counts all cluster-candidate pairs that satisfy the cuts.",
        "",
        "## Matching Statistics",
        "",
        f"- Total covered true clusters: `{stats['total_groups']}`",
        f"- Matched true clusters: `{stats['matched_groups']}`",
        f"- Match rate: `{match_rate:.2%}`",
        f"- Total candidates: `{stats['total_candidates']}`",
        f"- Candidates participating in at least one best match: `{stats['matched_candidates']}`",
        f"- Candidate participation fraction: `{cand_rate:.2%}`",
        f"- Matched pairs: `{stats['matched_pairs']}`",
        f"- Mean nearest projected distance: `{stats['mean_distance_mpc_h']:.4f} pMpc/h`",
        f"- Median nearest projected distance: `{stats['median_distance_mpc_h']:.4f} pMpc/h`",
        f"- Mean nearest `|Δz|`: `{stats['mean_abs_dz']:.4f}`",
        f"- Median nearest `|Δz|`: `{stats['median_abs_dz']:.4f}`",
        "",
        "## Brief Analysis",
        "",
        f"- Field {field_id} reaches a covered-cluster recovery rate of `{match_rate:.2%}` under the current blind-search candidate list and the adopted `{match_radius_mpc:.1f} pMpc/h + {dz_factor:.3f}(1+z)` cross-match rule.",
        f"- Only `{cand_rate:.2%}` of blind-search candidates participate in at least one valid pair, which means the candidate list is much denser than the true-cluster surface density inside the field footprint.",
        f"- The median projected offset of `{stats['median_distance_mpc_h']:.4f} pMpc/h` is more robust than the mean and is the cleaner indicator of positional agreement.",
        f"- The median `|Δz|` of `{stats['median_abs_dz']:.4f}` summarizes the typical redshift consistency between recovered true clusters and their nearest blind-search matches.",
        "",
        "## Output Files",
        "",
        f"- Result root: `{result_root}`",
        f"- Covered true-cluster table: `{summary_csv}`",
        f"- Match table: `{match_csv}`",
        f"- Sky diagnostic: `{plot_sky_path}`",
        f"- Offset diagnostic: `{plot_offset_path}`",
    ]

    if len(match_df) > 0:
        top = match_df.nsmallest(5, "distance_mpc_h")[["bcg_id", "candidate_id", "distance_mpc_h", "abs_dz"]]
        lines.extend(
            [
                "",
                "## Nearest Matched Pairs",
                "",
                "| bcg_id | candidate_id | distance_mpc_h | abs_dz |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in top.iterrows():
            lines.append(
                f"| {row['bcg_id']} | {int(row['candidate_id'])} | {row['distance_mpc_h']:.4f} | {row['abs_dz']:.4f} |"
            )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    field_id = args.field_id
    field_fits = Path(args.field_fits)
    candidate_csv = Path(args.candidate_csv)
    result_root = Path(args.result_root)
    summary_md = Path(args.summary_md)
    ensure_dirs(result_root, summary_md)

    summary_csv = result_root / f"field{field_id}_true_clusters_covered.csv"
    match_csv = result_root / f"field{field_id}_crossmatch_matches.csv"
    plot_sky_path = result_root / f"field{field_id}_crossmatch_sky.png"
    plot_offset_path = result_root / f"field{field_id}_crossmatch_offsets.png"

    field_df, polygon_xy, center_ra, center_dec = load_field_catalog(field_fits, args.buffer_deg)
    df_clusters = load_true_clusters(polygon_xy, field_df, center_ra, center_dec)
    df_cand = load_candidates(candidate_csv)
    match_df, stats = cross_match(df_clusters, df_cand, args.match_radius_mpc, args.dz_factor)

    df_clusters.to_csv(summary_csv, index=False)
    match_df.to_csv(match_csv, index=False)
    plot_sky(field_id, field_df, df_clusters, df_cand, polygon_xy, plot_sky_path)
    if len(match_df) > 0:
        plot_offsets(match_df, plot_offset_path)
    write_summary(
        field_id,
        field_fits,
        candidate_csv,
        result_root,
        summary_csv,
        match_csv,
        plot_sky_path,
        plot_offset_path,
        summary_md,
        stats,
        df_clusters,
        df_cand,
        match_df,
        args.match_radius_mpc,
        args.dz_factor,
    )

    print(f"Covered true clusters: {len(df_clusters)}")
    print(f"Candidates used: {len(df_cand)}")
    print(f"Matched true clusters: {stats['matched_groups']}")
    print(f"Match CSV: {match_csv}")
    print(f"Summary MD: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
