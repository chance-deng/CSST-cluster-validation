import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
import pandas as pd


VERSION_TAG = "version_1_1"
PROJECT_ROOT = "/Users/dengcanze/Documents/CSST"
CODE_DIR = os.path.join(PROJECT_ROOT, "Codex", "code")
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "Codex", "notebook")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "Codex", "result", VERSION_TAG)

BLIND_OUTPUT_DIR = os.path.join(RESULT_ROOT, "blind_search")
PPM_OUTPUT_DIR = os.path.join(RESULT_ROOT, "ppm_validation")
WORKFLOW_OUTPUT_DIR = os.path.join(RESULT_ROOT, "workflow")
LEGACY_RESULT_ROOT = os.path.join(PROJECT_ROOT, "Codex", "result", "version_1_0")
LEGACY_BLIND_OUTPUT_DIR = os.path.join(LEGACY_RESULT_ROOT, "blind_search")

MPL_CACHE_DIR = os.path.join(RESULT_ROOT, "mpl_cache")
PY_CACHE_DIR = os.path.join(RESULT_ROOT, "__pycache__")

REUSE_BLIND_RESULTS_IF_PRESENT = True
AUTO_BUILD_WORKFLOW = True
PPM_WORKERS = min(6, max(1, (os.cpu_count() or 2) - 1))
PPM_MAXTASKSPERCHILD = 20


def ensure_runtime_dirs():
    for path in [
        RESULT_ROOT,
        BLIND_OUTPUT_DIR,
        PPM_OUTPUT_DIR,
        WORKFLOW_OUTPUT_DIR,
        MPL_CACHE_DIR,
        PY_CACHE_DIR,
        NOTEBOOK_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


ensure_runtime_dirs()
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("PYTHONPYCACHEPREFIX", PY_CACHE_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")


import ppm_blindsearch_ppm_pipeline_v1_0 as core  # noqa: E402


def configure_core_paths():
    core.VERSION_TAG = VERSION_TAG
    core.PROJECT_ROOT = PROJECT_ROOT
    core.CODE_DIR = CODE_DIR
    core.NOTEBOOK_DIR = NOTEBOOK_DIR
    core.RESULT_ROOT = RESULT_ROOT
    core.BLIND_OUTPUT_DIR = BLIND_OUTPUT_DIR
    core.PPM_OUTPUT_DIR = PPM_OUTPUT_DIR
    core.WORKFLOW_OUTPUT_DIR = WORKFLOW_OUTPUT_DIR


def get_blind_csv_path():
    return os.path.join(BLIND_OUTPUT_DIR, "COSMOS_Web_PPM_Candidates_v1_0.csv")


def get_legacy_blind_csv_path():
    return os.path.join(LEGACY_BLIND_OUTPUT_DIR, "COSMOS_Web_PPM_Candidates_v1_0.csv")


def get_ppm_csv_path():
    return os.path.join(PPM_OUTPUT_DIR, "COSMOS_Web_PPM_Candidates_Results_v1_1.csv")


def get_final_candidate_csv_path():
    return os.path.join(RESULT_ROOT, "COSMOS_Web_BlindSearch_PPM_Final_Candidates_v1_1.csv")


def copy_legacy_blind_outputs():
    if not os.path.exists(LEGACY_BLIND_OUTPUT_DIR):
        return False

    os.makedirs(BLIND_OUTPUT_DIR, exist_ok=True)
    copied_any = False
    for name in os.listdir(LEGACY_BLIND_OUTPUT_DIR):
        src = os.path.join(LEGACY_BLIND_OUTPUT_DIR, name)
        dst = os.path.join(BLIND_OUTPUT_DIR, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied_any = True
    return copied_any


def load_or_run_blind_search():
    if REUSE_BLIND_RESULTS_IF_PRESENT:
        copy_legacy_blind_outputs()

    blind_csv_path = get_blind_csv_path()
    if REUSE_BLIND_RESULTS_IF_PRESENT and os.path.exists(blind_csv_path):
        print(f"Reusing existing blind-search table: {blind_csv_path}", flush=True)
        blind_df = pd.read_csv(blind_csv_path).sort_values("significance", ascending=False).reset_index(drop=True)
        top_candidate_id = int(blind_df.iloc[0]["ID"])
        return blind_df, top_candidate_id, blind_csv_path

    print("No reusable blind-search result found. Running blind search...", flush=True)
    return core.run_blind_search()


def dump_top_candidate_blind_density_slice(candidates_df, trace_candidate_id):
    top_row = candidates_df[candidates_df["ID"] == int(trace_candidate_id)].iloc[0]
    catalog = core.Table.read(core.BLIND_INPUT_CATALOG)
    ra_all = core.np.asarray(catalog["ra"].data)
    dec_all = core.np.asarray(catalog["dec"].data)
    z_all = core.np.asarray(catalog["zfinal"].data)
    finite_mask = core.np.isfinite(ra_all) & core.np.isfinite(dec_all) & core.np.isfinite(z_all)
    ra_all = ra_all[finite_mask]
    dec_all = dec_all[finite_mask]
    z_all = z_all[finite_mask]

    z_cen = float(top_row["z_peak"])
    dynamic_delta_z = 0.03 * (1 + z_cen)
    mask = (z_all >= z_cen - dynamic_delta_z) & (z_all <= z_cen + dynamic_delta_z)
    ra_slice = ra_all[mask]
    dec_slice = dec_all[mask]

    ra_bins = core.np.arange(core.np.min(ra_all), core.np.max(ra_all), core.PIXEL_SCALE)
    dec_bins = core.np.arange(core.np.min(dec_all), core.np.max(dec_all), core.PIXEL_SCALE)
    hist_2d, _, _ = core.np.histogram2d(ra_slice, dec_slice, bins=[ra_bins, dec_bins])

    scale_deg_per_mpc = core.cosmo.arcsec_per_kpc_proper(z_cen).to(core.u.deg / core.u.Mpc).value
    target_sigma_deg = float(top_row["best_scale_mpc"]) * scale_deg_per_mpc
    sigma_pixels = core.np.clip(target_sigma_deg / core.PIXEL_SCALE, 1.0, 15.0)

    smoothed_map = core.gaussian_filter(hist_2d, sigma=sigma_pixels, mode="constant")
    local_bg_map = core.gaussian_filter(hist_2d, sigma=sigma_pixels * 15, mode="constant")
    local_bg_map[local_bg_map <= 0] = 1e-5
    sig_map = (smoothed_map - local_bg_map) / core.np.sqrt(local_bg_map)
    delta_map = (smoothed_map - local_bg_map) / local_bg_map

    core.np.savez(
        os.path.join(BLIND_OUTPUT_DIR, "blind_density_slice_top_candidate_v1_1.npz"),
        candidate_id=int(trace_candidate_id),
        z_cen=z_cen,
        dynamic_delta_z=dynamic_delta_z,
        ra_candidate=float(top_row["RA"]),
        dec_candidate=float(top_row["Dec"]),
        best_scale_mpc=float(top_row["best_scale_mpc"]),
        ra_bins=ra_bins,
        dec_bins=dec_bins,
        hist_2d=hist_2d,
        smoothed_map=smoothed_map,
        local_bg_map=local_bg_map,
        sig_map=sig_map,
        delta_map=delta_map,
    )


def run_ppm_validation_stable(candidates_df, trace_candidate_id):
    print(f"Loading PPM background catalog: {core.PPM_BACKGROUND_CATALOG}", flush=True)
    bg_table = core.Table.read(core.PPM_BACKGROUND_CATALOG)
    valid_mask = core.np.isfinite(bg_table["zfinal"].data)
    bg_z = core.np.asarray(bg_table["zfinal"].data[valid_mask])
    bg_ra = core.np.asarray(bg_table["ra"].data[valid_mask])
    bg_dec = core.np.asarray(bg_table["dec"].data[valid_mask])
    bg_z_sorted = core.np.sort(bg_z)

    smg_coords = core.SkyCoord(
        ra=candidates_df["RA"].values * core.u.degree,
        dec=candidates_df["Dec"].values * core.u.degree,
    )
    bg_coords = core.SkyCoord(ra=bg_ra * core.u.degree, dec=bg_dec * core.u.degree)
    max_search_radius = core.np.sqrt(core.MAX_ANNULI * core.AREA_PER_ANNULUS / core.np.pi) * core.u.arcmin
    idx_smg, idx_bg, d2d, _ = core.search_around_sky(smg_coords, bg_coords, max_search_radius)

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

    ctx = mp.get_context("spawn")
    results_summary = []
    top_map_payload = None

    print(
        "Running stable PPM validation with "
        f"{PPM_WORKERS} spawn workers, maxtasksperchild={PPM_MAXTASKSPERCHILD}...",
        flush=True,
    )
    with ctx.Pool(
        processes=PPM_WORKERS,
        maxtasksperchild=PPM_MAXTASKSPERCHILD,
    ) as pool:
        for count, result in enumerate(pool.imap_unordered(core.run_ppm_worker, tasks, chunksize=1), start=1):
            candidate_id, best_cand, plot_payload, trace_payload = result

            plot_file = os.path.join(PPM_OUTPUT_DIR, f"PPM_candidate_{candidate_id}.png")
            core.save_ppm_scatter_plot(
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
                top_map_payload = plot_payload

            row = {"ID": candidate_id}
            if best_cand:
                row["PPM_z_mean"] = best_cand["z_mean"]
                row["PPM_z_rms"] = best_cand["z_rms"]
                row["PPM_rmin_mean"] = best_cand["rmin_mean"]
                row["PPM_rmax_mean"] = best_cand["rmax_mean"]
                row["PPM_richness"] = best_cand["richness"]
                row["PPM_significance"] = best_cand["significance"]
            else:
                row["PPM_z_mean"] = core.np.nan
                row["PPM_z_rms"] = core.np.nan
                row["PPM_rmin_mean"] = core.np.nan
                row["PPM_rmax_mean"] = core.np.nan
                row["PPM_richness"] = 0
                row["PPM_significance"] = core.np.nan
            results_summary.append(row)

            if count % 50 == 0 or count == len(tasks):
                print(f"  PPM progress: {count}/{len(tasks)}", flush=True)

    ppm_result_df = pd.DataFrame(results_summary)
    final_df = candidates_df.merge(ppm_result_df, on="ID", how="left")
    final_df = final_df.sort_values("significance", ascending=False).reset_index(drop=True)

    ppm_csv_path = get_ppm_csv_path()
    final_df.to_csv(ppm_csv_path, index=False)
    final_candidate_csv_path = get_final_candidate_csv_path()
    final_df.to_csv(final_candidate_csv_path, index=False)

    if top_map_payload is not None:
        np.savez(
            os.path.join(PPM_OUTPUT_DIR, "ppm_trace_map_top_candidate_v1_1.npz"),
            candidate_id=int(trace_candidate_id),
            beacon_z=float(top_map_payload["beacon_z"]),
            z_centroids=np.asarray(top_map_payload["z_centroids"]),
            delta_z_values=np.asarray(top_map_payload["delta_z_values"]),
            smoothed_significance_map=np.asarray(top_map_payload["smoothed_significance_map"]),
        )

    top_row = final_df[final_df["ID"] == int(trace_candidate_id)].iloc[0]
    summary_path = os.path.join(PPM_OUTPUT_DIR, "ppm_validation_summary_v1_1.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Version: {VERSION_TAG}\n")
        handle.write(f"Background catalog: {core.PPM_BACKGROUND_CATALOG}\n")
        handle.write(f"Candidates processed: {len(candidates_df)}\n")
        handle.write(f"PPM worker model: spawn, workers={PPM_WORKERS}, maxtasksperchild={PPM_MAXTASKSPERCHILD}\n")
        handle.write(f"Final merged candidate CSV: {final_candidate_csv_path}\n")
        handle.write(
            f"Trace candidate ID: {trace_candidate_id}, blind z_peak={top_row['z_peak']}, "
            f"PPM_z_mean={top_row['PPM_z_mean']}, PPM_significance={top_row['PPM_significance']}\n"
        )

    return final_df, ppm_csv_path, final_candidate_csv_path


def write_pipeline_note(blind_csv_path, ppm_csv_path, top_id):
    note_path = os.path.join(NOTEBOOK_DIR, "ppm_blindsearch_pipeline_v1_1.md")
    with open(note_path, "w", encoding="utf-8") as handle:
        handle.write("# PPM Blind Search + Validation Pipeline v1.1\n\n")
        handle.write(
            "- This version is hardened for macOS by using a non-GUI matplotlib backend and "
            "running PPM multiprocessing with `spawn` instead of `fork`.\n"
        )
        handle.write(
            "- All PPM figures are rendered in the main process, so later workflow steps no longer "
            "need manual plot rebuilding.\n"
        )
        handle.write(f"- Main script: `{os.path.join(CODE_DIR, 'ppm_blindsearch_ppm_pipeline_v1_1.py')}`\n")
        handle.write(f"- Workflow script: `{os.path.join(CODE_DIR, 'plot_top_candidate_workflow_v1_1.py')}`\n")
        handle.write(f"- Blind-search output CSV: `{blind_csv_path}`\n")
        handle.write(f"- PPM validation output CSV: `{ppm_csv_path}`\n")
        handle.write(f"- Final merged candidate CSV: `{get_final_candidate_csv_path()}`\n")
        handle.write(f"- Top candidate ID used for workflow tracing: `{top_id}`\n")
        handle.write(f"- Result root: `{RESULT_ROOT}`\n")


def build_workflow_figure():
    if not AUTO_BUILD_WORKFLOW:
        return

    import plot_top_candidate_workflow_v1_1 as workflow

    workflow.main()


def main():
    configure_core_paths()
    core.ensure_directories()

    blind_final_df, top_candidate_id, blind_csv_path = load_or_run_blind_search()
    dump_top_candidate_blind_density_slice(blind_final_df, top_candidate_id)
    ppm_final_df, ppm_csv_path, final_candidate_csv_path = run_ppm_validation_stable(
        blind_final_df,
        top_candidate_id,
    )

    pipeline_summary = os.path.join(RESULT_ROOT, "pipeline_summary_v1_1.txt")
    top_row = ppm_final_df[ppm_final_df["ID"] == int(top_candidate_id)].iloc[0]
    with open(pipeline_summary, "w", encoding="utf-8") as handle:
        handle.write(f"Version: {VERSION_TAG}\n")
        handle.write(f"Blind-search input: {core.BLIND_INPUT_CATALOG}\n")
        handle.write(f"PPM background catalog: {core.PPM_BACKGROUND_CATALOG}\n")
        handle.write(f"Blind-search candidates: {len(blind_final_df)}\n")
        handle.write(f"Top candidate ID: {top_candidate_id}\n")
        handle.write(f"Top candidate blind significance: {top_row['significance']:.3f}\n")
        handle.write(f"Top candidate PPM significance: {top_row['PPM_significance']}\n")
        handle.write(f"Blind-search CSV: {blind_csv_path}\n")
        handle.write(f"PPM validation CSV: {ppm_csv_path}\n")
        handle.write(f"Final merged candidate CSV: {final_candidate_csv_path}\n")
        handle.write(f"Workflow output directory: {WORKFLOW_OUTPUT_DIR}\n")

    write_pipeline_note(blind_csv_path, ppm_csv_path, top_candidate_id)
    build_workflow_figure()

    print("\nStable macOS pipeline v1.1 finished successfully.", flush=True)
    print(f"Blind-search CSV: {blind_csv_path}", flush=True)
    print(f"PPM validation CSV: {ppm_csv_path}", flush=True)
    print(f"Final merged candidate CSV: {final_candidate_csv_path}", flush=True)
    print(f"Workflow directory: {WORKFLOW_OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, CODE_DIR)
    main()
