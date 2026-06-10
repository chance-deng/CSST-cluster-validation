import multiprocessing as mp
import os
import warnings
from itertools import groupby

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.special import erfinv
from scipy.stats import poisson


warnings.filterwarnings("ignore")

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


# ===================== Version 1.0 Configuration =====================
VERSION_TAG = "version_1_0"
PROJECT_ROOT = "/Users/dengcanze/Documents/CSST"
CODE_DIR = os.path.join(PROJECT_ROOT, "Codex", "code")
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "Codex", "notebook")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "Codex", "result", VERSION_TAG)

BLIND_OUTPUT_DIR = os.path.join(RESULT_ROOT, "blind_search")
PPM_OUTPUT_DIR = os.path.join(RESULT_ROOT, "ppm_validation")
WORKFLOW_OUTPUT_DIR = os.path.join(RESULT_ROOT, "workflow")

BLIND_INPUT_CATALOG = os.path.join(
    PROJECT_ROOT,
    "SMG_trace_ov_code",
    "data",
    "filtered_galaxies_mask_all_easy.fits",
)
PPM_BACKGROUND_CATALOG = (
    "/Users/dengcanze/Documents/SMG_trace_overdensity/"
    "Data/COSMOS2025_photo_z_catalog_type_galaxies.fits"
)


# --------------------- Blind Search Parameters ---------------------
PIXEL_SCALE = 0.3 / 60.0
Z_MIN, Z_MAX = 0.1, 3.8
Z_STEP = 0.01
SIGMA_THRESHOLD = 0.2
DELTA_THRESHOLD = 0.5
TARGET_SCALES_MPC = [0.4, 0.8, 1.2]
DBSCAN_SPATIAL_EPS = 2.0 / 60.0
MIN_GALAXIES_PER_SLICE = 5
MERGE_SPATIAL_MPC_TH = 1.0
MERGE_DZ_FACTOR = 0.04
MIN_DET_SLICES = 2
MERGE_Z_BIN = 0.02
MERGE_CELL_SIZE_DEG = 0.2
MERGE_PROGRESS_EVERY = 20000


# --------------------- PPM Parameters ---------------------
BG_AREA_ARCMIN2 = 1976.8
MAX_ANNULI = 50
AREA_PER_ANNULUS = 3.14
PPM_TARGET_DZ = 0.295
PPM_TARGET_DZ_TOL = 0.005
PPM_MAX_Z = 8.0
PPM_N_Z = 800
PPM_DZ_MIN = 0.02
PPM_DZ_MAX = 0.4
PPM_N_DZ = 50
PPM_INTERVAL_START_S = 2.0
PPM_INTERVAL_DS = 0.1
PPM_INTERVAL_MAX_ITER = 50
PPM_BEST_MATCH_DZ = 0.3


def ensure_directories():
    for path in [RESULT_ROOT, BLIND_OUTPUT_DIR, PPM_OUTPUT_DIR, WORKFLOW_OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)


def build_cosmo_lut(z_min, z_max, z_step):
    z_arr = np.arange(z_min, z_max + z_step, z_step)
    scale = cosmo.arcsec_per_kpc_proper(z_arr).to(u.deg / u.Mpc).value
    da = cosmo.angular_diameter_distance(z_arr).value
    return {round(float(z), 4): (float(s), float(d)) for z, s, d in zip(z_arr, scale, da)}


def extract_peaks_from_slice(
    ra_gal,
    dec_gal,
    z_cen,
    ra_bins,
    dec_bins,
    target_scales=None,
):
    if target_scales is None:
        target_scales = TARGET_SCALES_MPC

    scale_deg_per_mpc = cosmo.arcsec_per_kpc_proper(z_cen).to(u.deg / u.Mpc).value
    hist_2d, _, _ = np.histogram2d(ra_gal, dec_gal, bins=[ra_bins, dec_bins])
    slice_candidates = []

    for target_physical_mpc in target_scales:
        target_sigma_deg = target_physical_mpc * scale_deg_per_mpc
        sigma_pixels = np.clip(target_sigma_deg / PIXEL_SCALE, 1.0, 15.0)

        smoothed_map = gaussian_filter(hist_2d, sigma=sigma_pixels, mode="constant")
        local_bg_map = gaussian_filter(hist_2d, sigma=sigma_pixels * 15, mode="constant")
        local_bg_map[local_bg_map <= 0] = 1e-5

        sig_map = (smoothed_map - local_bg_map) / np.sqrt(local_bg_map)
        delta_map = (smoothed_map - local_bg_map) / local_bg_map

        local_max = maximum_filter(sig_map, size=5) == sig_map
        peaks_mask = (
            local_max
            & (sig_map >= SIGMA_THRESHOLD)
            & (delta_map >= DELTA_THRESHOLD)
        )
        peak_idx = np.where(peaks_mask)
        search_radius_deg = target_sigma_deg * 1.5

        for ix, iy in zip(peak_idx[0], peak_idx[1]):
            grid_ra = ra_bins[ix] + PIXEL_SCALE / 2
            grid_dec = dec_bins[iy] + PIXEL_SCALE / 2

            dist_sq = (ra_gal - grid_ra) ** 2 + (dec_gal - grid_dec) ** 2
            near_mask = dist_sq < search_radius_deg**2
            if np.sum(near_mask) > 0:
                refined_ra = float(np.average(ra_gal[near_mask]))
                refined_dec = float(np.average(dec_gal[near_mask]))
                n_members = int(np.sum(near_mask))
            else:
                refined_ra = float(grid_ra)
                refined_dec = float(grid_dec)
                n_members = 0

            slice_candidates.append(
                {
                    "RA": refined_ra,
                    "Dec": refined_dec,
                    "z_cen": float(z_cen),
                    "significance": float(sig_map[ix, iy]),
                    "delta": float(delta_map[ix, iy]),
                    "scale_mpc": float(target_physical_mpc),
                    "sigma_pixels": float(sigma_pixels),
                    "n_members": n_members,
                }
            )

    return slice_candidates


def merge_candidates_greedy_nms(candidates_df, spatial_mpc_th=1.0, dz_factor=0.04):
    if len(candidates_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = candidates_df.sort_values("significance", ascending=False).reset_index(drop=True)
    raw_ids = df_sorted["raw_id"].to_numpy(dtype=np.int64)
    ra_vals = df_sorted["RA"].to_numpy(dtype=np.float64)
    dec_vals = df_sorted["Dec"].to_numpy(dtype=np.float64)
    z_vals = df_sorted["z_cen"].to_numpy(dtype=np.float64)
    sig_vals = df_sorted["significance"].to_numpy(dtype=np.float64)
    delta_vals = df_sorted["delta"].to_numpy(dtype=np.float64)
    scale_vals = df_sorted["scale_mpc"].to_numpy(dtype=np.float64)
    members_vals = df_sorted["n_members"].to_numpy(dtype=np.int64)

    kept_candidates = []
    assignment_rows = []
    spatial_index = {}
    max_merge_dz = dz_factor * (1 + np.max(z_vals)) + Z_STEP
    z_bin_span = int(np.ceil(max_merge_dz / MERGE_Z_BIN))

    def cluster_key(z_cen, ra, dec):
        return (
            int(np.floor(z_cen / MERGE_Z_BIN)),
            int(np.floor(ra / MERGE_CELL_SIZE_DEG)),
            int(np.floor(dec / MERGE_CELL_SIZE_DEG)),
        )

    def register_cluster(cluster_index, z_cen, ra, dec):
        z_bin, ra_bin, dec_bin = cluster_key(z_cen, ra, dec)
        spatial_index.setdefault((z_bin, ra_bin, dec_bin), []).append(cluster_index)

    total_candidates = len(df_sorted)
    for idx_row in range(total_candidates):
        cand_ra = ra_vals[idx_row]
        cand_dec = dec_vals[idx_row]
        cand_z = z_vals[idx_row]
        cand_sig = sig_vals[idx_row]
        cand_delta = delta_vals[idx_row]
        cand_scale = scale_vals[idx_row]
        cand_members = members_vals[idx_row]
        cand_raw_id = raw_ids[idx_row]
        cand_dict = {
            "raw_id": int(cand_raw_id),
            "RA": float(cand_ra),
            "Dec": float(cand_dec),
            "z_cen": float(cand_z),
            "significance": float(cand_sig),
            "delta": float(cand_delta),
            "scale_mpc": float(cand_scale),
            "n_members": int(cand_members),
        }

        if len(kept_candidates) == 0:
            cluster_index = 0
            cand_dict["det_unique_slices"] = {cand_z}
            cand_dict["detected_scales"] = {cand_scale}
            cand_dict["z_min_det"] = cand_z
            cand_dict["z_max_det"] = cand_z
            cand_dict["_da_mpc"] = float(cosmo.angular_diameter_distance(cand_z).value)
            cand_dict["_cos_dec"] = float(np.cos(np.radians(cand_dec)))
            kept_candidates.append(cand_dict)
            register_cluster(cluster_index, cand_z, cand_ra, cand_dec)
            assignment_rows.append(
                {"raw_id": int(cand_raw_id), "cluster_index": cluster_index, "merge_type": "seed"}
            )
            continue

        z_bin, ra_bin, dec_bin = cluster_key(cand_z, cand_ra, cand_dec)
        candidate_clusters = set()
        for zb in range(z_bin - z_bin_span, z_bin + z_bin_span + 1):
            for rb in range(ra_bin - 1, ra_bin + 2):
                for db in range(dec_bin - 1, dec_bin + 2):
                    candidate_clusters.update(spatial_index.get((zb, rb, db), []))

        if not candidate_clusters:
            cluster_index = len(kept_candidates)
            cand_dict["det_unique_slices"] = {cand_z}
            cand_dict["detected_scales"] = {cand_scale}
            cand_dict["z_min_det"] = cand_z
            cand_dict["z_max_det"] = cand_z
            cand_dict["_da_mpc"] = float(cosmo.angular_diameter_distance(cand_z).value)
            cand_dict["_cos_dec"] = float(np.cos(np.radians(cand_dec)))
            kept_candidates.append(cand_dict)
            register_cluster(cluster_index, cand_z, cand_ra, cand_dec)
            assignment_rows.append(
                {"raw_id": int(cand_raw_id), "cluster_index": cluster_index, "merge_type": "new_z"}
            )
            continue

        conflict = False
        chosen_cluster = None
        for idx in sorted(candidate_clusters):
            kept = kept_candidates[idx]
            dz_threshold = dz_factor * (1 + kept["z_cen"])
            if abs(cand_z - kept["z_cen"]) > dz_threshold:
                continue

            d_ra = np.radians(cand_ra - kept["RA"]) * kept["_cos_dec"]
            d_dec = np.radians(cand_dec - kept["Dec"])
            sep_rad = np.sqrt(d_ra**2 + d_dec**2)
            dist_mpc = (sep_rad * kept["_da_mpc"]) * 0.7

            if dist_mpc <= spatial_mpc_th:
                conflict = True
                chosen_cluster = idx
                kept["det_unique_slices"].add(cand_z)
                kept["detected_scales"].add(cand_scale)
                kept["z_min_det"] = min(kept["z_min_det"], cand_z)
                kept["z_max_det"] = max(kept["z_max_det"], cand_z)
                kept["n_members"] = max(int(kept["n_members"]), int(cand_members))
                break

        if not conflict:
            cluster_index = len(kept_candidates)
            cand_dict["det_unique_slices"] = {cand_z}
            cand_dict["detected_scales"] = {cand_scale}
            cand_dict["z_min_det"] = cand_z
            cand_dict["z_max_det"] = cand_z
            cand_dict["_da_mpc"] = float(cosmo.angular_diameter_distance(cand_z).value)
            cand_dict["_cos_dec"] = float(np.cos(np.radians(cand_dec)))
            kept_candidates.append(cand_dict)
            register_cluster(cluster_index, cand_z, cand_ra, cand_dec)
            assignment_rows.append(
                {
                    "raw_id": int(cand_raw_id),
                    "cluster_index": cluster_index,
                    "merge_type": "new_spatial",
                }
            )
        else:
            assignment_rows.append(
                {
                    "raw_id": int(cand_raw_id),
                    "cluster_index": int(chosen_cluster),
                    "merge_type": "merged",
                }
            )

        if (idx_row + 1) % MERGE_PROGRESS_EVERY == 0 or idx_row + 1 == total_candidates:
            print(
                f"  merge {idx_row + 1}/{total_candidates}: "
                f"kept_clusters={len(kept_candidates)}",
                flush=True,
            )

    final_list = []
    cluster_to_id = {}
    running_id = 1
    for cluster_index, cluster in enumerate(kept_candidates):
        unique_slices_count = len(cluster["det_unique_slices"])
        if unique_slices_count < MIN_DET_SLICES:
            continue

        cluster_to_id[cluster_index] = running_id
        z_peak = float(cluster["z_cen"])
        final_list.append(
            {
                "ID": running_id,
                "RA": round(float(cluster["RA"]), 6),
                "Dec": round(float(cluster["Dec"]), 6),
                "z_cen": round(z_peak, 3),
                "significance": round(float(cluster["significance"]), 3),
                "scale_mpc": round(float(cluster["scale_mpc"]), 3),
                "det_unique_slices": unique_slices_count,
                "z_min_det": round(float(cluster["z_min_det"]), 3),
                "z_max_det": round(float(cluster["z_max_det"]), 3),
                "z_peak": round(z_peak, 3),
                "z_range": f"{cluster['z_min_det']:.2f}-{cluster['z_max_det']:.2f}",
                "best_scale_mpc": round(float(max(cluster["detected_scales"])), 3),
                "delta": round(float(cluster["delta"]), 3),
                "scales_hit": "|".join(map(str, sorted(cluster["detected_scales"]))),
                "n_members": int(cluster["n_members"]),
            }
        )
        running_id += 1

    assignment_df = pd.DataFrame(assignment_rows)
    if len(assignment_df) > 0:
        assignment_df["ID"] = assignment_df["cluster_index"].map(cluster_to_id)
        assignment_df["survived"] = assignment_df["ID"].notna()

    final_df = pd.DataFrame(final_list).sort_values("significance", ascending=False).reset_index(drop=True)
    return final_df, assignment_df


def run_blind_search():
    print(f"Loading blind-search catalog: {BLIND_INPUT_CATALOG}")
    catalog = Table.read(BLIND_INPUT_CATALOG)
    ra_all = np.asarray(catalog["ra"].data)
    dec_all = np.asarray(catalog["dec"].data)
    z_all = np.asarray(catalog["zfinal"].data)

    finite_mask = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all)
    ra_all = np.asarray(ra_all[finite_mask], dtype=np.float32)
    dec_all = np.asarray(dec_all[finite_mask], dtype=np.float32)
    z_all = np.asarray(z_all[finite_mask], dtype=np.float32)

    # Sort once in redshift so each slice can use a bounded window rather than
    # scanning the full catalog with a fresh boolean mask.
    order = np.argsort(z_all)
    z_all = z_all[order]
    ra_all = ra_all[order]
    dec_all = dec_all[order]

    ra_bins = np.arange(np.min(ra_all), np.max(ra_all), PIXEL_SCALE)
    dec_bins = np.arange(np.min(dec_all), np.max(dec_all), PIXEL_SCALE)
    z_centroids = np.arange(Z_MIN, Z_MAX, Z_STEP)

    all_raw_candidates = []
    slice_rows = []

    print("[1/4] Running blind search slice scan...")
    total_slices = len(z_centroids)
    for i, z_cen in enumerate(z_centroids, start=1):
        dynamic_delta_z = 0.03 * (1 + z_cen)
        z_lo = z_cen - dynamic_delta_z
        z_hi = z_cen + dynamic_delta_z
        lo = np.searchsorted(z_all, z_lo, side="left")
        hi = np.searchsorted(z_all, z_hi, side="right")
        n_galaxies = int(hi - lo)

        if n_galaxies < MIN_GALAXIES_PER_SLICE:
            slice_rows.append(
                {
                    "z_cen": round(float(z_cen), 3),
                    "dynamic_delta_z": round(float(dynamic_delta_z), 4),
                    "n_galaxies": n_galaxies,
                    "n_peaks": 0,
                }
            )
            if i == 1 or i % 20 == 0 or i == total_slices:
                print(
                    f"  slice {i}/{total_slices}: z={z_cen:.3f}, "
                    f"dz={dynamic_delta_z:.4f}, galaxies={n_galaxies}, peaks=0",
                    flush=True,
                )
            continue

        peaks = extract_peaks_from_slice(
            ra_all[lo:hi],
            dec_all[lo:hi],
            z_cen,
            ra_bins,
            dec_bins,
            target_scales=TARGET_SCALES_MPC,
        )
        for peak in peaks:
            peak["slice_delta_z"] = float(dynamic_delta_z)
        all_raw_candidates.extend(peaks)
        slice_rows.append(
            {
                "z_cen": round(float(z_cen), 3),
                "dynamic_delta_z": round(float(dynamic_delta_z), 4),
                "n_galaxies": n_galaxies,
                "n_peaks": len(peaks),
            }
        )
        if i == 1 or i % 20 == 0 or i == total_slices:
            print(
                f"  slice {i}/{total_slices}: z={z_cen:.3f}, "
                f"dz={dynamic_delta_z:.4f}, galaxies={n_galaxies}, peaks={len(peaks)}",
                flush=True,
            )

    raw_df = pd.DataFrame(all_raw_candidates)
    if len(raw_df) == 0:
        raise RuntimeError("Blind search found no raw candidates.")

    raw_df.insert(0, "raw_id", np.arange(1, len(raw_df) + 1))
    slice_df = pd.DataFrame(slice_rows)

    print(f"[2/4] Raw peak extraction complete: {len(raw_df)} peaks")
    print("[3/4] Running Greedy NMS candidate merging...")
    final_df, assignment_df = merge_candidates_greedy_nms(
        raw_df,
        spatial_mpc_th=MERGE_SPATIAL_MPC_TH,
        dz_factor=MERGE_DZ_FACTOR,
    )
    if len(final_df) == 0:
        raise RuntimeError("No blind-search candidates survived the persistence filter.")

    final_df = final_df.sort_values("significance", ascending=False).reset_index(drop=True)

    blind_raw_path = os.path.join(BLIND_OUTPUT_DIR, "blind_raw_candidates_v1_0.csv")
    blind_slice_path = os.path.join(BLIND_OUTPUT_DIR, "blind_slice_scan_stats_v1_0.csv")
    blind_merged_path = os.path.join(BLIND_OUTPUT_DIR, "blind_merge_assignments_v1_0.csv")
    blind_final_path = os.path.join(BLIND_OUTPUT_DIR, "COSMOS_Web_PPM_Candidates_v1_0.csv")

    raw_df.to_csv(blind_raw_path, index=False)
    slice_df.to_csv(blind_slice_path, index=False)
    assignment_df.to_csv(blind_merged_path, index=False)
    final_df.to_csv(blind_final_path, index=False)

    top_candidate = final_df.iloc[0]
    top_id = int(top_candidate["ID"])
    top_trace = raw_df.merge(assignment_df[["raw_id", "ID", "merge_type", "survived"]], on="raw_id", how="left")
    top_trace = top_trace[top_trace["ID"] == top_id].sort_values(["z_cen", "scale_mpc"]).reset_index(drop=True)
    top_trace_path = os.path.join(BLIND_OUTPUT_DIR, "blind_top_candidate_trace_v1_0.csv")
    top_trace.to_csv(top_trace_path, index=False)

    blind_summary_txt = os.path.join(BLIND_OUTPUT_DIR, "blind_search_summary_v1_0.txt")
    with open(blind_summary_txt, "w", encoding="utf-8") as handle:
        handle.write(f"Version: {VERSION_TAG}\n")
        handle.write(f"Input catalog: {BLIND_INPUT_CATALOG}\n")
        handle.write(f"Total galaxies used: {len(ra_all)}\n")
        handle.write(f"Total z slices scanned: {len(z_centroids)}\n")
        handle.write(f"Raw peaks found: {len(raw_df)}\n")
        handle.write(f"Final candidates kept: {len(final_df)}\n")
        handle.write(
            "Top candidate: "
            f"ID={top_id}, RA={top_candidate['RA']}, Dec={top_candidate['Dec']}, "
            f"z_peak={top_candidate['z_peak']}, significance={top_candidate['significance']}\n"
        )

    print(f"[4/4] Blind search complete: {len(final_df)} candidates kept")
    return final_df, top_id, blind_final_path


def create_annuli(n_annuli=50, area_per_annulus=3.14):
    return np.sqrt(np.arange(1, n_annuli + 1) * area_per_annulus / np.pi)


def reduce_to_1d(significance_map, rmin_map, rmax_map, z_centroids, delta_z_values):
    dz_mask = (
        (delta_z_values >= PPM_TARGET_DZ - PPM_TARGET_DZ_TOL)
        & (delta_z_values <= PPM_TARGET_DZ + PPM_TARGET_DZ_TOL)
    )
    significance_1d = np.nanmean(significance_map[:, dz_mask], axis=1)
    rmin_1d = np.nanmean(rmin_map[:, dz_mask], axis=1)
    rmax_1d = np.nanmean(rmax_map[:, dz_mask], axis=1)
    return z_centroids, significance_1d, rmin_1d, rmax_1d


def find_s_intervals(z_centroids, significance, s_threshold, dz_min_sep=0.02, dz_min_length=0.05):
    above_threshold = significance >= s_threshold
    intervals = []
    start_idx = None
    for idx, is_above in enumerate(above_threshold):
        if is_above and start_idx is None:
            start_idx = idx
        elif (not is_above) and start_idx is not None:
            intervals.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        intervals.append((start_idx, len(above_threshold) - 1))

    merged_intervals = []
    if intervals:
        current_start, current_end = intervals[0]
        for idx in range(1, len(intervals)):
            start, end = intervals[idx]
            gap = z_centroids[start] - z_centroids[current_end]
            if gap <= dz_min_sep:
                current_end = end
            else:
                if z_centroids[current_end] - z_centroids[current_start] >= dz_min_length:
                    merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        if z_centroids[current_end] - z_centroids[current_start] >= dz_min_length:
            merged_intervals.append((current_start, current_end))
    return merged_intervals


def iterative_peak_finding(z_centroids, significance, rmin, rmax):
    all_intervals = {}
    current_s = PPM_INTERVAL_START_S
    max_s_intervals = []

    for _ in range(PPM_INTERVAL_MAX_ITER):
        intervals = find_s_intervals(z_centroids, significance, current_s)
        if intervals:
            all_intervals[current_s] = intervals
            max_s_intervals = intervals
        else:
            break
        current_s += PPM_INTERVAL_DS

    final_intervals = []
    if not all_intervals:
        return final_intervals

    s_values = sorted(all_intervals.keys(), reverse=True)
    for interval in max_s_intervals:
        idx_start, idx_end = interval
        final_intervals.append(
            {
                "z_range": (z_centroids[idx_start], z_centroids[idx_end]),
                "significance": float(np.mean(significance[idx_start : idx_end + 1])),
                "rmin": float(np.mean(rmin[idx_start : idx_end + 1])),
                "rmax": float(np.mean(rmax[idx_start : idx_end + 1])),
                "threshold": float(s_values[0]),
            }
        )

    for idx in range(1, len(s_values)):
        current_s = s_values[idx]
        higher_s = s_values[idx - 1]
        for curr_interval in all_intervals[current_s]:
            curr_z_start = z_centroids[curr_interval[0]]
            curr_z_end = z_centroids[curr_interval[1]]
            is_covered = any(
                (
                    curr_z_start >= z_centroids[high[0]]
                    and curr_z_end <= z_centroids[high[1]]
                )
                for high in all_intervals[higher_s]
            )
            if not is_covered:
                final_intervals.append(
                    {
                        "z_range": (curr_z_start, curr_z_end),
                        "significance": float(
                            np.mean(significance[curr_interval[0] : curr_interval[1] + 1])
                        ),
                        "rmin": float(np.mean(rmin[curr_interval[0] : curr_interval[1] + 1])),
                        "rmax": float(np.mean(rmax[curr_interval[0] : curr_interval[1] + 1])),
                        "threshold": float(current_s),
                    }
                )

    unique_intervals = []
    seen_ranges = set()
    for interval in final_intervals:
        rounded_range = (
            round(float(interval["z_range"][0]), 3),
            round(float(interval["z_range"][1]), 3),
        )
        if rounded_range not in seen_ranges:
            seen_ranges.add(rounded_range)
            unique_intervals.append(interval)
    return unique_intervals


def merge_nearby_intervals_simple(intervals, z_threshold=0.005):
    if not intervals:
        return []

    for interval in intervals:
        interval["z_center"] = round(sum(interval["z_range"]) / 2, 4)

    sorted_intervals = sorted(intervals, key=lambda item: item["significance"], reverse=True)
    merged_intervals = []
    for interval in sorted_intervals:
        should_merge = False
        for idx, merged in enumerate(merged_intervals):
            if abs(interval["z_center"] - merged["z_center"]) <= z_threshold:
                should_merge = True
                if interval["significance"] > merged["significance"]:
                    merged_intervals[idx] = interval
                break
        if not should_merge:
            merged_intervals.append(interval)
    return merged_intervals


def estimate_cluster_properties(final_intervals, z_centroids_1d, local_z, local_dist, rmin_1d, rmax_1d):
    cluster_props = []
    for interval in final_intervals:
        z_start, z_end = interval["z_range"]
        z_center = (z_start + z_end) / 2
        in_interval = (z_centroids_1d >= z_start) & (z_centroids_1d <= z_end)

        s_rmin = rmin_1d[in_interval]
        s_rmax = rmax_1d[in_interval]
        rmin_mean = float(np.mean(s_rmin)) if len(s_rmin) > 0 else 0.0
        rmin_rms = float(np.std(s_rmin)) if len(s_rmin) > 0 else 0.0
        rmax_mean = float(np.mean(s_rmax)) if len(s_rmax) > 0 else 0.0
        rmax_rms = float(np.std(s_rmax)) if len(s_rmax) > 0 else 0.0

        z_min_cluster = z_center - PPM_TARGET_DZ / 2
        z_max_cluster = z_center + PPM_TARGET_DZ / 2
        z_mask = (local_z >= z_min_cluster) & (local_z <= z_max_cluster)
        selected_z = local_z[z_mask]
        selected_dist = local_dist[z_mask]

        in_radius_range = (selected_dist >= rmin_mean) & (selected_dist <= rmax_mean)
        final_z = selected_z[in_radius_range]

        richness = len(final_z)
        z_mean = float(np.mean(final_z)) if richness > 0 else float(z_center)
        z_rms = float(np.std(final_z)) if richness > 0 else 0.08

        cluster_props.append(
            {
                "z_center": float(z_center),
                "z_mean": z_mean,
                "z_rms": z_rms,
                "rmin_mean": rmin_mean,
                "rmin_rms": rmin_rms,
                "rmax_mean": rmax_mean,
                "rmax_rms": rmax_rms,
                "richness": int(richness),
                "significance": float(interval["significance"]),
            }
        )
    return cluster_props


def save_ppm_text_result(result_file, candidate_id, ra, dec, z_beacon, cluster_props):
    with open(result_file, "w", encoding="utf-8") as handle:
        handle.write(f"Candidate ID: {candidate_id}\n")
        handle.write(f"RA: {ra:.6f}\n")
        handle.write(f"Dec: {dec:.6f}\n")
        handle.write(f"Blind-search z_peak: {z_beacon:.3f}\n\n")
        handle.write("PPM candidates:\n")
        if cluster_props:
            for idx, prop in enumerate(cluster_props, start=1):
                handle.write(f"\nCandidate {idx}\n")
                handle.write(f"  z_mean = {prop['z_mean']:.3f}\n")
                handle.write(f"  z_rms = {prop['z_rms']:.3f}\n")
                handle.write(f"  rmin_mean = {prop['rmin_mean']:.2f} arcmin\n")
                handle.write(f"  rmax_mean = {prop['rmax_mean']:.2f} arcmin\n")
                handle.write(f"  richness = {prop['richness']}\n")
                handle.write(f"  significance = {prop['significance']:.2f} sigma\n")
        else:
            handle.write("No significant PPM candidate found.\n")


def save_ppm_scatter_plot(output_path, smoothed_significance_map, z_centroids, delta_z_values, beacon_z):
    plot_sig = smoothed_significance_map.copy()
    plot_sig[plot_sig < 2] = np.nan

    z_grid, delta_grid = np.meshgrid(z_centroids, delta_z_values)
    values = plot_sig.T

    z_flat = z_grid.flatten()
    dz_flat = delta_grid.flatten()
    vals_flat = values.flatten()
    valid_mask = ~np.isnan(vals_flat)

    z_data = z_flat[valid_mask]
    dz_data = dz_flat[valid_mask]
    val_data = vals_flat[valid_mask]

    colors = ["cyan", "green", "blue", "red", "brown", "black"]
    bounds = [2, 3, 4, 5, 6, 7, 20]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    for idx in range(len(bounds) - 1):
        lower = bounds[idx]
        upper = bounds[idx + 1]
        level_mask = (val_data >= lower) & (val_data < upper)
        ax.scatter(
            z_data[level_mask],
            dz_data[level_mask],
            color=colors[idx],
            s=35,
            marker="o",
            edgecolors="none",
            linewidths=0.5,
        )

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    ax.set_xlim(z_centroids.min(), z_centroids.max() + 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"z$_{centroid}$", fontsize=26)
    ax.set_ylabel(r"$\Delta z$", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.axvline(
        beacon_z,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label=rf"z$_{{blind}}$={beacon_z:.3f}",
    )
    ax.legend(fontsize=18, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_ppm_worker(args):
    (
        candidate_id,
        beacon_ra,
        beacon_dec,
        beacon_z,
        local_dist,
        local_z,
        bg_z_sorted,
        trace_candidate_id,
        ppm_output_dir,
    ) = args

    annuli_radii = create_annuli(MAX_ANNULI, AREA_PER_ANNULUS)
    annuli_edges = np.concatenate(([0], annuli_radii))
    z_centroids = np.linspace(0, PPM_MAX_Z, PPM_N_Z)
    delta_z_values = np.linspace(PPM_DZ_MIN, PPM_DZ_MAX, PPM_N_DZ)

    significance_map = np.zeros((len(z_centroids), len(delta_z_values)), dtype=np.float32)
    rmin_map = np.zeros_like(significance_map)
    rmax_map = np.zeros_like(significance_map)

    for i, z_cen in enumerate(z_centroids):
        for j, delta_z in enumerate(delta_z_values):
            z_min = z_cen - delta_z / 2
            z_max = z_cen + delta_z / 2

            idx_min = np.searchsorted(bg_z_sorted, z_min)
            idx_max = np.searchsorted(bg_z_sorted, z_max)
            bg_density = (idx_max - idx_min) / BG_AREA_ARCMIN2
            if bg_density <= 0:
                continue

            mask = (local_z >= z_min) & (local_z <= z_max)
            dist_in_z = local_dist[mask]
            if len(dist_in_z) == 0:
                continue

            counts, _ = np.histogram(dist_in_z, bins=annuli_edges)
            lambda_per_annulus = bg_density * AREA_PER_ANNULUS
            probabilities = np.ones(len(counts))
            nonzero = counts > 0
            probabilities[nonzero] = 1 - poisson.cdf(counts[nonzero] - 1, lambda_per_annulus)

            significant = probabilities < 0.3
            regions = []
            for key, values in groupby(enumerate(significant), key=lambda item: item[1]):
                if key:
                    regions.append([entry[0] for entry in values])

            first_region = regions[0] if regions else []
            if not first_region:
                continue

            current_min = 0 if first_region[0] == 0 else annuli_radii[first_region[0] - 1]
            current_max = annuli_radii[first_region[-1]]
            if current_min * 60 >= 132:
                continue

            total_count = np.sum(counts[first_region])
            lambda_merged = bg_density * len(first_region) * AREA_PER_ANNULUS
            prob_merged = (
                1 - poisson.cdf(total_count - 1, lambda_merged) if total_count > 0 else 1.0
            )
            sigma = 8.0 if prob_merged < 1e-10 else abs(np.sqrt(2) * erfinv(1 - 2 * prob_merged))

            significance_map[i, j] = sigma
            rmin_map[i, j] = current_min
            rmax_map[i, j] = current_max

    dz_pixel = PPM_MAX_Z / (PPM_N_Z - 1)
    dz_delta_pixel = (PPM_DZ_MAX - PPM_DZ_MIN) / (PPM_N_DZ - 1)
    smoothed_significance_map = gaussian_filter(
        significance_map,
        sigma=(0.02 / dz_pixel, 0.02 / dz_delta_pixel),
    )

    z_centroids_1d, significance_1d, rmin_1d, rmax_1d = reduce_to_1d(
        smoothed_significance_map,
        rmin_map,
        rmax_map,
        z_centroids,
        delta_z_values,
    )
    intervals = iterative_peak_finding(z_centroids_1d, significance_1d, rmin_1d, rmax_1d)
    intervals = merge_nearby_intervals_simple(intervals)
    cluster_props = estimate_cluster_properties(
        intervals,
        z_centroids_1d,
        local_z,
        local_dist,
        rmin_1d,
        rmax_1d,
    )

    result_file = os.path.join(ppm_output_dir, f"PPM_candidate_{candidate_id}.txt")
    save_ppm_text_result(result_file, candidate_id, beacon_ra, beacon_dec, beacon_z, cluster_props)

    best_valid_candidate = None
    if cluster_props:
        valid_candidates = [
            prop for prop in cluster_props if abs(prop["z_mean"] - beacon_z) <= PPM_BEST_MATCH_DZ
        ]
        if valid_candidates:
            best_valid_candidate = sorted(
                valid_candidates,
                key=lambda item: item["significance"],
                reverse=True,
            )[0]

    interval_rows = [
        {
            "z_start": item["z_range"][0],
            "z_end": item["z_range"][1],
            "z_center": item["z_center"],
            "significance": item["significance"],
            "rmin": item["rmin"],
            "rmax": item["rmax"],
            "threshold": item["threshold"],
        }
        for item in intervals
    ]

    trace_payload = None
    if int(candidate_id) == int(trace_candidate_id):
        trace_payload = {
            "curve_df": pd.DataFrame(
                {
                    "z_centroid": z_centroids_1d,
                    "significance_1d": significance_1d,
                    "rmin_1d": rmin_1d,
                    "rmax_1d": rmax_1d,
                }
            ),
            "intervals_df": pd.DataFrame(interval_rows),
        }

    plot_payload = {
        "candidate_id": int(candidate_id),
        "beacon_z": float(beacon_z),
        "z_centroids": z_centroids,
        "delta_z_values": delta_z_values,
        "smoothed_significance_map": smoothed_significance_map,
    }

    return candidate_id, best_valid_candidate, plot_payload, trace_payload


def run_ppm_validation(candidates_df, trace_candidate_id):
    print(f"Loading PPM background catalog: {PPM_BACKGROUND_CATALOG}")
    bg_table = Table.read(PPM_BACKGROUND_CATALOG)
    valid_mask = np.isfinite(bg_table["zfinal"].data)
    bg_z = np.asarray(bg_table["zfinal"].data[valid_mask])
    bg_ra = np.asarray(bg_table["ra"].data[valid_mask])
    bg_dec = np.asarray(bg_table["dec"].data[valid_mask])
    bg_z_sorted = np.sort(bg_z)

    smg_coords = SkyCoord(
        ra=candidates_df["RA"].values * u.degree,
        dec=candidates_df["Dec"].values * u.degree,
    )
    bg_coords = SkyCoord(ra=bg_ra * u.degree, dec=bg_dec * u.degree)
    max_search_radius = np.sqrt(MAX_ANNULI * AREA_PER_ANNULUS / np.pi) * u.arcmin
    idx_smg, idx_bg, d2d, _ = search_around_sky(smg_coords, bg_coords, max_search_radius)

    tasks = []
    for idx in range(len(candidates_df)):
        row = candidates_df.iloc[idx]
        local_mask = idx_smg == idx
        local_dist = d2d[local_mask].arcmin
        local_z = bg_z[idx_bg[local_mask]]
        tasks.append(
            (
                int(row["ID"]),
                float(row["RA"]),
                float(row["Dec"]),
                float(row["z_peak"]),
                local_dist,
                local_z,
                bg_z_sorted,
                int(trace_candidate_id),
                PPM_OUTPUT_DIR,
            )
        )

    print("Running PPM validation...")
    results = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for result in pool.imap_unordered(run_ppm_worker, tasks):
            results.append(result)

    result_rows = []
    for candidate_id, best_cand, plot_payload, trace_payload in results:
        plot_file = os.path.join(PPM_OUTPUT_DIR, f"PPM_candidate_{candidate_id}.png")
        save_ppm_scatter_plot(
            plot_file,
            plot_payload["smoothed_significance_map"],
            plot_payload["z_centroids"],
            plot_payload["delta_z_values"],
            plot_payload["beacon_z"],
        )

        if trace_payload is not None:
            trace_payload["curve_df"].to_csv(
                os.path.join(PPM_OUTPUT_DIR, "ppm_trace_curve_top_candidate_v1_0.csv"),
                index=False,
            )
            trace_payload["intervals_df"].to_csv(
                os.path.join(PPM_OUTPUT_DIR, "ppm_trace_intervals_top_candidate_v1_0.csv"),
                index=False,
            )

        row = {"ID": candidate_id}
        if best_cand:
            row["PPM_z_mean"] = best_cand["z_mean"]
            row["PPM_z_rms"] = best_cand["z_rms"]
            row["PPM_rmin_mean"] = best_cand["rmin_mean"]
            row["PPM_rmax_mean"] = best_cand["rmax_mean"]
            row["PPM_richness"] = best_cand["richness"]
            row["PPM_significance"] = best_cand["significance"]
        else:
            row["PPM_z_mean"] = np.nan
            row["PPM_z_rms"] = np.nan
            row["PPM_rmin_mean"] = np.nan
            row["PPM_rmax_mean"] = np.nan
            row["PPM_richness"] = 0
            row["PPM_significance"] = np.nan
        result_rows.append(row)

    ppm_result_df = pd.DataFrame(result_rows)
    final_df = candidates_df.merge(ppm_result_df, on="ID", how="left")
    final_df = final_df.sort_values("significance", ascending=False).reset_index(drop=True)

    ppm_csv_path = os.path.join(PPM_OUTPUT_DIR, "COSMOS_Web_PPM_Candidates_Results_v1_0.csv")
    final_df.to_csv(ppm_csv_path, index=False)

    top_row = final_df[final_df["ID"] == int(trace_candidate_id)].iloc[0]
    summary_path = os.path.join(PPM_OUTPUT_DIR, "ppm_validation_summary_v1_0.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Version: {VERSION_TAG}\n")
        handle.write(f"Background catalog: {PPM_BACKGROUND_CATALOG}\n")
        handle.write(f"Candidates processed: {len(candidates_df)}\n")
        handle.write(
            f"Trace candidate ID: {trace_candidate_id}, blind z_peak={top_row['z_peak']}, "
            f"PPM_z_mean={top_row['PPM_z_mean']}, PPM_significance={top_row['PPM_significance']}\n"
        )

    return final_df, ppm_csv_path


def write_pipeline_note(blind_csv_path, ppm_csv_path, top_id):
    note_path = os.path.join(NOTEBOOK_DIR, "ppm_blindsearch_pipeline_v1_0.md")
    with open(note_path, "w", encoding="utf-8") as handle:
        handle.write("# PPM Blind Search + Validation Pipeline v1.0\n\n")
        handle.write(f"- Main script: `{os.path.join(CODE_DIR, 'ppm_blindsearch_ppm_pipeline_v1_0.py')}`\n")
        handle.write(
            f"- Workflow plot script: `{os.path.join(CODE_DIR, 'plot_top_candidate_workflow_v1_0.py')}`\n"
        )
        handle.write(f"- Blind-search output CSV: `{blind_csv_path}`\n")
        handle.write(f"- PPM validation output CSV: `{ppm_csv_path}`\n")
        handle.write(f"- Top candidate ID traced for workflow plotting: `{top_id}`\n")
        handle.write(
            f"- Result root: `{RESULT_ROOT}`\n"
        )


def main():
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    ensure_directories()
    _ = build_cosmo_lut(Z_MIN, Z_MAX, Z_STEP)

    blind_final_df, top_candidate_id, blind_csv_path = run_blind_search()
    ppm_final_df, ppm_csv_path = run_ppm_validation(blind_final_df, top_candidate_id)

    pipeline_summary = os.path.join(RESULT_ROOT, "pipeline_summary_v1_0.txt")
    top_row = ppm_final_df[ppm_final_df["ID"] == int(top_candidate_id)].iloc[0]
    with open(pipeline_summary, "w", encoding="utf-8") as handle:
        handle.write(f"Version: {VERSION_TAG}\n")
        handle.write(f"Blind-search input: {BLIND_INPUT_CATALOG}\n")
        handle.write(f"PPM background catalog: {PPM_BACKGROUND_CATALOG}\n")
        handle.write(f"Blind-search candidates: {len(blind_final_df)}\n")
        handle.write(f"Top candidate ID: {top_candidate_id}\n")
        handle.write(
            f"Top candidate blind significance: {top_row['significance']:.3f}\n"
        )
        handle.write(
            f"Top candidate PPM significance: {top_row['PPM_significance']}\n"
        )
        handle.write(f"Blind-search CSV: {blind_csv_path}\n")
        handle.write(f"PPM validation CSV: {ppm_csv_path}\n")
        handle.write(f"Workflow output directory: {WORKFLOW_OUTPUT_DIR}\n")

    write_pipeline_note(blind_csv_path, ppm_csv_path, top_candidate_id)

    print("\nPipeline finished successfully.")
    print(f"Blind-search CSV: {blind_csv_path}")
    print(f"PPM validation CSV: {ppm_csv_path}")
    print(
        "Next step: run "
        f"{os.path.join(CODE_DIR, 'plot_top_candidate_workflow_v1_0.py')} "
        "to create the 4:3 workflow figure."
    )


if __name__ == "__main__":
    main()
