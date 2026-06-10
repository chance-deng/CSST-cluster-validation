#!/usr/bin/env python3
"""Run multithreaded PPM for i<22, n_members>=7 candidates and matched true clusters."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
import astropy.units as u
from scipy.spatial import ConvexHull

import ppm_blindsearch_ppm_pipeline_v1_0 as core


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"

CANDIDATE_CSV = RESULT_ROOT / "version_1_1_noppm_field05_7band_i_lt_22p0/blind_search/COSMOS_Web_PPM_Candidates_v1_0.csv"
MATCH_CSV = RESULT_ROOT / "field05_i_cut_grid_crossmatch_r1p5_thresholds/i_lt_22p0/cross_match_r1p5/field05_crossmatch_matches.csv"
SEVENBAND_BG = RESULT_ROOT / "blindsearch_inputs/field05_7band_i_band_cuts/field05_7band_i_lt_22p0_blindsearch_input.fits"
TRUE_BG = RESULT_ROOT / "galaxies_C6_field05_blindsearch_input.fits"

OUTROOT = RESULT_ROOT / "field05_i22_nmembers7_ppm_candidates_and_true_clusters"
SUMMARY_CAND_CSV = OUTROOT / "candidate_ppm_summary.csv"
SUMMARY_TRUE_CSV = OUTROOT / "true_cluster_ppm_summary.csv"
MASTER_NOTE = NOTEBOOK_ROOT / "field05_i22_nmembers7_ppm_candidates_and_true_clusters.md"

NMEMBERS_MIN = 7

_BG_Z_SORTED: np.ndarray | None = None
_BG_AREA_ARCMIN2: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--skip-clusters", action="store_true")
    return parser.parse_args()


def compute_field_area_arcmin2(field_fits: Path) -> float:
    tab = Table.read(field_fits, memmap=True)
    ra = np.asarray(tab["ra"], dtype=float)
    dec = np.asarray(tab["dec"], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[mask]
    dec = dec[mask]
    center_ra = float(np.median(ra))
    center_dec = float(np.median(dec))
    x = (ra - center_ra) * np.cos(np.radians(center_dec))
    y = dec - center_dec
    area_deg2 = float(ConvexHull(np.column_stack([x, y])).volume)
    return area_deg2 * 3600.0


def init_worker(bg_area_arcmin2: float, bg_z_sorted: np.ndarray) -> None:
    global _BG_Z_SORTED, _BG_AREA_ARCMIN2
    _BG_Z_SORTED = bg_z_sorted
    _BG_AREA_ARCMIN2 = bg_area_arcmin2
    core.BG_AREA_ARCMIN2 = float(bg_area_arcmin2)


def ppm_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _BG_Z_SORTED is None or _BG_AREA_ARCMIN2 is None:
        raise RuntimeError("PPM worker was not initialized")
    core.BG_AREA_ARCMIN2 = float(_BG_AREA_ARCMIN2)

    ppm_dir = Path(task["ppm_dir"])
    ppm_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(task["plot_path"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_id, best_cand, plot_payload, _ = core.run_ppm_worker(
        (
            int(task["ppm_id"]),
            float(task["ra"]),
            float(task["dec"]),
            float(task["z_peak"]),
            np.asarray(task["local_dist"], dtype=float),
            np.asarray(task["local_z"], dtype=float),
            _BG_Z_SORTED,
            -1,
            str(ppm_dir),
        )
    )
    core.save_ppm_scatter_plot(
        str(plot_path),
        plot_payload["smoothed_significance_map"],
        plot_payload["z_centroids"],
        plot_payload["delta_z_values"],
        plot_payload["beacon_z"],
    )

    out = dict(task["meta"])
    out.update(
        {
            "ppm_id": int(candidate_id),
            "plot_path": str(plot_path),
            "ppm_text_path": str(ppm_dir / f"PPM_candidate_{int(candidate_id)}.txt"),
            "ppm_cluster_count": 0 if best_cand is None else 1,
        }
    )
    if best_cand:
        out.update(
            {
                "PPM_z_mean": best_cand["z_mean"],
                "PPM_z_rms": best_cand["z_rms"],
                "PPM_rmin_mean": best_cand["rmin_mean"],
                "PPM_rmax_mean": best_cand["rmax_mean"],
                "PPM_richness": best_cand["richness"],
                "PPM_significance": best_cand["significance"],
            }
        )
    else:
        out.update(
            {
                "PPM_z_mean": np.nan,
                "PPM_z_rms": np.nan,
                "PPM_rmin_mean": np.nan,
                "PPM_rmax_mean": np.nan,
                "PPM_richness": np.nan,
                "PPM_significance": np.nan,
            }
        )
    return out


def selected_candidates(max_candidates: int | None) -> pd.DataFrame:
    cand = pd.read_csv(CANDIDATE_CSV)
    cand = cand[pd.to_numeric(cand["n_members"], errors="coerce") >= NMEMBERS_MIN].copy()
    cand = cand.sort_values(["n_members", "significance"], ascending=[False, False]).reset_index(drop=True)
    if max_candidates is not None:
        cand = cand.head(max_candidates).copy()
    return cand


def selected_true_clusters(selected_cand: pd.DataFrame, max_clusters: int | None) -> pd.DataFrame:
    matches = pd.read_csv(MATCH_CSV)
    selected_ids = set(selected_cand["ID"].astype(int))
    matches = matches[matches["candidate_id"].astype(int).isin(selected_ids)].copy()
    matches = matches.sort_values(["distance_mpc_h", "abs_dz"]).drop_duplicates(subset=["cluster_index"]).reset_index(drop=True)
    if max_clusters is not None:
        matches = matches.head(max_clusters).copy()
    return matches


def load_background(bg_fits: Path, z_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    area = compute_field_area_arcmin2(bg_fits)
    tab = Table.read(bg_fits, memmap=True)
    z = np.asarray(tab[z_col], dtype=float)
    ra = np.asarray(tab["ra"], dtype=float)
    dec = np.asarray(tab["dec"], dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    return ra[ok], dec[ok], z[ok], np.sort(z[ok]), area


def build_tasks(beacons: pd.DataFrame, bg_fits: Path, z_col: str, kind: str) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    bg_ra, bg_dec, bg_z, bg_z_sorted, area = load_background(bg_fits, z_col)
    beacon_coords = SkyCoord(ra=beacons["RA"].to_numpy(float) * u.deg, dec=beacons["Dec"].to_numpy(float) * u.deg)
    bg_coords = SkyCoord(ra=bg_ra * u.deg, dec=bg_dec * u.deg)
    max_search_radius = np.sqrt(core.MAX_ANNULI * core.AREA_PER_ANNULUS / np.pi) * u.arcmin
    idx_beacon, idx_bg, d2d, _ = search_around_sky(beacon_coords, bg_coords, max_search_radius)

    tasks: list[dict[str, Any]] = []
    for idx, row in beacons.iterrows():
        local_mask = idx_beacon == idx
        local_dist = d2d[local_mask].arcmin.astype(float)
        local_z = bg_z[idx_bg[local_mask]].astype(float)
        ppm_id = int(row["ppm_id"])
        ppm_dir = OUTROOT / "ppm_outputs" / kind / f"{kind}_{idx + 1:05d}_ppm{ppm_id}"
        plot_path = OUTROOT / "plots" / kind / f"{kind}_{idx + 1:05d}_ppm{ppm_id}.png"
        tasks.append(
            {
                "ppm_id": ppm_id,
                "ra": float(row["RA"]),
                "dec": float(row["Dec"]),
                "z_peak": float(row["z_peak"]),
                "local_dist": local_dist,
                "local_z": local_z,
                "ppm_dir": str(ppm_dir),
                "plot_path": str(plot_path),
                "meta": row.to_dict(),
            }
        )
    return tasks, bg_z_sorted, area


def run_tasks(tasks: list[dict[str, Any]], bg_z_sorted: np.ndarray, area: float, workers: int, label: str) -> pd.DataFrame:
    rows = []
    print(f"Running {label} PPM: tasks={len(tasks)}, workers={workers}, area={area:.3f} arcmin^2", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(float(area), bg_z_sorted)) as ex:
        futures = [ex.submit(ppm_worker, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i == 1 or i % 50 == 0 or i == len(futures):
                print(f"  {label}: {i}/{len(futures)} complete", flush=True)
    return pd.DataFrame(rows)


def build_candidate_beacons(cand: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ppm_id": 410000000 + cand["ID"].astype(int),
            "candidate_id": cand["ID"].astype(int),
            "RA": cand["RA"].astype(float),
            "Dec": cand["Dec"].astype(float),
            "z_peak": cand["z_peak"].astype(float),
            "blind_significance": cand["significance"].astype(float),
            "n_members": cand["n_members"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def build_true_cluster_beacons(matches: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ppm_id": 420000000 + matches["cluster_index"].astype(int),
            "cluster_index": matches["cluster_index"].astype(int),
            "bcg_id": matches["bcg_id"].astype(str),
            "matched_candidate_id": matches["candidate_id"].astype(int),
            "RA": matches["cluster_ra"].astype(float),
            "Dec": matches["cluster_dec"].astype(float),
            "z_peak": matches["cluster_z"].astype(float),
            "candidate_z_peak": matches["candidate_z_peak"].astype(float),
            "distance_mpc_h": matches["distance_mpc_h"].astype(float),
            "abs_dz": matches["abs_dz"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def write_note(cand_summary: pd.DataFrame | None, true_summary: pd.DataFrame | None) -> None:
    lines = [
        "# Field05 i<22 n_members>=7 PPM",
        "",
        f"- Candidate selection: `i < 22.0`, `n_members >= {NMEMBERS_MIN}`",
        f"- Candidate background catalog: `{SEVENBAND_BG}`",
        f"- True-cluster background catalog: `{TRUE_BG}`",
        f"- Output root: `{OUTROOT}`",
        "",
    ]
    if cand_summary is not None:
        lines += [
            "## Candidate PPM",
            "",
            f"- Summary CSV: `{SUMMARY_CAND_CSV}`",
            f"- N: `{len(cand_summary)}`",
            f"- Median PPM significance: `{cand_summary['PPM_significance'].median():.4f}`",
            "",
        ]
    if true_summary is not None:
        lines += [
            "## Matched True-cluster PPM",
            "",
            f"- Summary CSV: `{SUMMARY_TRUE_CSV}`",
            f"- N: `{len(true_summary)}`",
            f"- Median PPM significance: `{true_summary['PPM_significance'].median():.4f}`",
            "",
        ]
    MASTER_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    cand = selected_candidates(args.max_candidates)
    matches = selected_true_clusters(cand, args.max_clusters)
    print(f"Selected candidates: {len(cand)}", flush=True)
    print(f"Matched true clusters: {len(matches)}", flush=True)

    cand_summary = None
    true_summary = None

    if not args.skip_candidates:
        cand_beacons = build_candidate_beacons(cand)
        cand_tasks, cand_bg_z_sorted, cand_area = build_tasks(cand_beacons, SEVENBAND_BG, "zfinal", "candidate")
        cand_summary = run_tasks(cand_tasks, cand_bg_z_sorted, cand_area, args.workers, "candidate")
        cand_summary.to_csv(SUMMARY_CAND_CSV, index=False)
        print(f"Wrote candidate PPM summary: {SUMMARY_CAND_CSV}", flush=True)

    if not args.skip_clusters:
        true_beacons = build_true_cluster_beacons(matches)
        true_tasks, true_bg_z_sorted, true_area = build_tasks(true_beacons, TRUE_BG, "redshift", "true_cluster")
        true_summary = run_tasks(true_tasks, true_bg_z_sorted, true_area, args.workers, "true_cluster")
        true_summary.to_csv(SUMMARY_TRUE_CSV, index=False)
        print(f"Wrote true-cluster PPM summary: {SUMMARY_TRUE_CSV}", flush=True)

    write_note(cand_summary, true_summary)
    print(f"Wrote note: {MASTER_NOTE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
