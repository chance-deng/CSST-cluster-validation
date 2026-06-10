#!/usr/bin/env python3
"""Cross-match full-7band i-cut candidates using tight field coverage masks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
CODE_DIR = PROJECT_ROOT / "Codex/code"
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"

FIELD_SUMMARY = RESULT_ROOT / "blindsearch_inputs/full7band_cross_lstm_i_mag_5fields/full7band_cross_lstm_5field_summary.csv"
FIELDS_01_04_GRID = RESULT_ROOT / "blindsearch_grid_full7band_fields01_04_i_cuts_v11/blindsearch_i_cut_grid_summary.csv"
OUTPUT_ROOT = RESULT_ROOT / "full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage"
SUMMARY_MD = NOTEBOOK_ROOT / "full7band_fields01_05_i_cut_blindsearch_crossmatch_grid_tightcoverage.md"

I_LIMITS = [21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
NMEMBER_THRESHOLDS = [3, 5, 7, 10, 15, 20, 30, 50, 80, 100]
MATCH_RADIUS_MPC_H = 1.5


def label_from_limit(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def fmt_int(value: float | int) -> str:
    return "-" if pd.isna(value) else f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def fmt_float(value: float, digits: int = 4) -> str:
    return "-" if pd.isna(value) else f"{float(value):.{digits}f}"


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


def candidate_csv_field05(i_limit: float) -> Path:
    label = label_from_limit(i_limit)
    return RESULT_ROOT / f"version_1_1_noppm_field05_7band_i_lt_{label}/blind_search/COSMOS_Web_PPM_Candidates_v1_0.csv"


def input_fits_field05(i_limit: float) -> Path:
    label = label_from_limit(i_limit)
    return RESULT_ROOT / f"blindsearch_inputs/field05_7band_i_band_cuts/field05_7band_i_lt_{label}_blindsearch_input.fits"


def load_jobs() -> list[dict]:
    grid = pd.read_csv(FIELDS_01_04_GRID)
    jobs: list[dict] = []
    for _, row in grid.iterrows():
        jobs.append(
            {
                "field_id": int(row["field_id"]),
                "i_limit": float(row["i_limit"]),
                "input_fits": Path(row["input_fits"]),
                "candidate_csv": Path(row["blind_csv"]),
                "input_rows_total": int(row["input_rows_total"]),
                "input_rows_kept": int(row["input_rows_kept"]),
            }
        )
    for i_limit in I_LIMITS:
        jobs.append(
            {
                "field_id": 5,
                "i_limit": float(i_limit),
                "input_fits": input_fits_field05(i_limit),
                "candidate_csv": candidate_csv_field05(i_limit),
                "input_rows_total": np.nan,
                "input_rows_kept": np.nan,
            }
        )
    return jobs


def build_tight_covered_clusters() -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    sys.path.insert(0, str(CODE_DIR))
    import check_full7band_irregular_field_coverage as coverage
    import reanalyze_field05_v11_crossmatch_r1p5_and_thresholds as base

    field_summary = pd.read_csv(FIELD_SUMMARY)
    cluster_all = Table.read(coverage.CLUSTER_FITS).to_pandas()
    cluster_all = cluster_all[np.isfinite(cluster_all["ra"]) & np.isfinite(cluster_all["dec"]) & np.isfinite(cluster_all["redshift"])].copy().reset_index(drop=True)

    cluster_by_field: dict[int, pd.DataFrame] = {}
    coverage_rows: list[dict] = []
    for _, frow in field_summary.iterrows():
        field_id = int(frow["field_id"])
        field_df = coverage.load_field(Path(frow["output_fits"]))
        center_ra = float(np.median(field_df["ra"]))
        center_dec = float(np.median(field_df["dec"]))
        gx = (field_df["ra"].to_numpy(float) - center_ra) * np.cos(np.radians(center_dec))
        gy = field_df["dec"].to_numpy(float) - center_dec
        cx = (cluster_all["ra"].to_numpy(float) - center_ra) * np.cos(np.radians(center_dec))
        cy = cluster_all["dec"].to_numpy(float) - center_dec

        mask, xmin, ymin, _, _ = coverage.make_grid_mask(gx, gy)
        tight = coverage.points_in_grid_mask(cx, cy, mask, xmin, ymin)
        covered = cluster_all[tight].copy().reset_index(drop=True)
        covered["cluster_index"] = np.arange(len(covered), dtype=int)
        cluster_by_field[field_id] = covered

        # Keep the previous convex count only for comparison in the note.
        polygon, _, _ = base.build_footprint_polygon(
            field_df["ra"].to_numpy(float),
            field_df["dec"].to_numpy(float),
            coverage.CONVEX_BUFFER_DEG,
        )
        current = base.points_in_polygon(np.column_stack([cx, cy]), polygon)
        coverage_rows.append(
            {
                "field_id": field_id,
                "full7band_galaxies": len(field_df),
                "convex_covered_true_clusters": int(current.sum()),
                "tight_covered_true_clusters": int(len(covered)),
                "removed_convex_only_clusters": int(current.sum() - len(covered)),
                "removed_fraction_vs_convex": float((current.sum() - len(covered)) / current.sum()) if current.sum() else np.nan,
            }
        )
    coverage_summary = pd.DataFrame(coverage_rows)
    return cluster_by_field, coverage_summary


def evaluate_threshold(base, pairs: pd.DataFrame, total_clusters: int, candidates: pd.DataFrame, threshold: int) -> dict:
    kept = candidates[candidates["n_members"] >= threshold].copy()
    kept_ids = set(kept["ID"].astype(int).tolist())
    use_pairs = pairs[pairs["candidate_id"].isin(kept_ids)].copy() if len(pairs) else pairs.copy()
    stats = base.summarize_matches(use_pairs, total_clusters, len(kept))
    matched_clusters = int(stats["matched_groups"])
    participating = int(stats["matched_candidates"])
    return {
        "n_members_threshold": int(threshold),
        "candidates_kept": int(len(kept)),
        "matched_true_clusters": matched_clusters,
        "match_rate": matched_clusters / total_clusters if total_clusters else np.nan,
        "participating_candidates": participating,
        "purity_proxy": participating / len(kept) if len(kept) else np.nan,
        "candidate_density": len(kept) / total_clusters if total_clusters else np.nan,
        "matched_pairs": int(stats["matched_pairs"]),
        "median_distance_mpc_h": stats["median_distance_mpc_h"],
        "median_abs_dz": stats["median_abs_dz"],
    }


def run_crossmatch_grid() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(CODE_DIR))
    import reanalyze_field05_v11_crossmatch_r1p5_and_thresholds as base

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cluster_by_field, coverage_summary = build_tight_covered_clusters()
    coverage_summary.to_csv(OUTPUT_ROOT / "tight_coverage_true_cluster_counts.csv", index=False)

    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    jobs = load_jobs()
    for n_job, job in enumerate(jobs, start=1):
        field_id = int(job["field_id"])
        i_limit = float(job["i_limit"])
        i_label = label_from_limit(i_limit)
        out_dir = OUTPUT_ROOT / f"field{field_id:02d}_i_lt_{i_label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        match_csv = out_dir / f"field{field_id:02d}_crossmatch_matches.csv"
        covered_csv = out_dir / f"field{field_id:02d}_true_clusters_covered_tight.csv"

        print(f"[{n_job:02d}/{len(jobs)}] field {field_id:02d}, i<{i_limit:g}")
        clusters = cluster_by_field[field_id]
        candidates = base.load_candidates(Path(job["candidate_csv"]))
        pairs = base.build_valid_pairs(clusters, candidates, MATCH_RADIUS_MPC_H)
        pairs = base.compute_offset_columns(pairs)
        stats = base.summarize_matches(pairs, len(clusters), len(candidates))

        clusters.to_csv(covered_csv, index=False)
        pairs.to_csv(match_csv, index=False)
        input_rows_kept = job["input_rows_kept"]
        if pd.isna(input_rows_kept):
            input_rows_kept = len(Table.read(Path(job["input_fits"])))

        row = {
            "field_id": field_id,
            "i_limit": i_limit,
            "input_rows_total": job["input_rows_total"],
            "input_rows_kept": int(input_rows_kept),
            "covered_true_clusters": int(stats["total_groups"]),
            "candidates": int(stats["total_candidates"]),
            "matched_true_clusters": int(stats["matched_groups"]),
            "match_rate": stats["matched_groups"] / stats["total_groups"] if stats["total_groups"] else np.nan,
            "participating_candidates": int(stats["matched_candidates"]),
            "purity_proxy": stats["matched_candidates"] / stats["total_candidates"] if stats["total_candidates"] else np.nan,
            "candidate_density": stats["total_candidates"] / stats["total_groups"] if stats["total_groups"] else np.nan,
            "matched_pairs": int(stats["matched_pairs"]),
            "median_distance_mpc_h": stats["median_distance_mpc_h"],
            "median_abs_dz": stats["median_abs_dz"],
            "match_csv": str(match_csv),
            "covered_cluster_csv": str(covered_csv),
            "candidate_csv": str(job["candidate_csv"]),
            "input_fits": str(job["input_fits"]),
        }
        summary_rows.append(row)
        for thr in NMEMBER_THRESHOLDS:
            trow = evaluate_threshold(base, pairs, len(clusters), candidates, thr)
            trow.update({"field_id": field_id, "i_limit": i_limit})
            threshold_rows.append(trow)
        print(
            "    tight_covered={covered:,} cand={cand:,} matched={matched:,} "
            "match={match:.2%} purity={purity:.2%}".format(
                covered=row["covered_true_clusters"],
                cand=row["candidates"],
                matched=row["matched_true_clusters"],
                match=row["match_rate"],
                purity=row["purity_proxy"],
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(["field_id", "i_limit"]).reset_index(drop=True)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["field_id", "i_limit", "n_members_threshold"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_ROOT / "all_fields_i_cut_crossmatch_summary_tightcoverage.csv", index=False)
    thresholds.to_csv(OUTPUT_ROOT / "all_fields_i_cut_nmembers_crossmatch_summary_tightcoverage.csv", index=False)
    return summary, thresholds, coverage_summary


def plot_metric_grid(summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    setup_style()
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B", 4: "#B279A2", 5: "#E45756"}
    metrics = [("match_rate", "Match rate"), ("purity_proxy", "Purity proxy"), ("candidate_density", "Candidate density")]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), dpi=300)
    for ax, (col, ylabel) in zip(axes, metrics):
        for field_id, sub in summary.groupby("field_id"):
            ax.plot(sub["i_limit"], sub[col], marker="o", lw=2.0, ms=4.5, color=colors[int(field_id)], label=f"Field {int(field_id):02d}")
        ax.set_xlabel(r"$i$-band cut")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.28)
        if col != "candidate_density":
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=9, frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / "all_fields_i_cut_crossmatch_metrics_tightcoverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for field_id in sorted(thresholds["field_id"].unique()):
        sub = thresholds[(thresholds["field_id"] == field_id) & (np.isclose(thresholds["i_limit"], 22.0))].copy()
        fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=300)
        ax2 = ax1.twinx()
        ax1.plot(sub["n_members_threshold"], sub["match_rate"], marker="o", lw=2.0, color="#1f77b4", label="Match rate")
        ax1.plot(sub["n_members_threshold"], sub["purity_proxy"], marker="s", lw=2.0, color="#d62728", label="Purity proxy")
        ax2.plot(sub["n_members_threshold"], sub["candidates_kept"], marker="^", lw=1.8, ls="--", color="#2ca02c", label="Candidates kept")
        ax1.set_xlabel(r"$n_{\rm members}$ threshold")
        ax1.set_ylabel("Fraction")
        ax2.set_ylabel("Candidates kept")
        ax1.set_ylim(0, 1.02)
        ax1.grid(True, ls="--", alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], fontsize=9, frameon=True, loc="center right")
        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"field{int(field_id):02d}_i_lt_22p0_nmembers_crossmatch_metrics_tightcoverage.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def table_baseline(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| field | i cut | covered true clusters | candidates | matched true clusters | match rate | purity proxy | candidate density | matched pairs | median d_proj | median abs dz |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | <{row['i_limit']:g} | {fmt_int(row['covered_true_clusters'])} | "
            f"{fmt_int(row['candidates'])} | {fmt_int(row['matched_true_clusters'])} | {fmt_pct(row['match_rate'])} | "
            f"{fmt_pct(row['purity_proxy'])} | {fmt_float(row['candidate_density'], 2)} | {fmt_int(row['matched_pairs'])} | "
            f"{fmt_float(row['median_distance_mpc_h'])} | {fmt_float(row['median_abs_dz'])} |"
        )
    return lines


def table_coverage(coverage_summary: pd.DataFrame) -> list[str]:
    lines = [
        "| field | full7band galaxies | convex covered | tight covered | removed convex-only | removed fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in coverage_summary.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | {fmt_int(row['full7band_galaxies'])} | "
            f"{fmt_int(row['convex_covered_true_clusters'])} | {fmt_int(row['tight_covered_true_clusters'])} | "
            f"{fmt_int(row['removed_convex_only_clusters'])} | {fmt_pct(row['removed_fraction_vs_convex'])} |"
        )
    return lines


def table_best_by_field(summary: pd.DataFrame, thresholds: pd.DataFrame) -> list[str]:
    lines = [
        "| field | baseline best i cut | baseline match rate | baseline purity | best F1 condition | candidates kept | matched true clusters | match rate | purity proxy | F1 |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for field_id in sorted(summary["field_id"].unique()):
        base_sub = summary[summary["field_id"] == field_id].copy()
        base_sub["f1"] = 2 * base_sub["match_rate"] * base_sub["purity_proxy"] / (base_sub["match_rate"] + base_sub["purity_proxy"]).replace(0, np.nan)
        best_base = base_sub.sort_values(["f1", "match_rate", "purity_proxy"], ascending=False).iloc[0]
        tsub = thresholds[thresholds["field_id"] == field_id].copy()
        tsub["f1"] = 2 * tsub["match_rate"] * tsub["purity_proxy"] / (tsub["match_rate"] + tsub["purity_proxy"]).replace(0, np.nan)
        best_thr = tsub.sort_values(["f1", "match_rate", "purity_proxy"], ascending=False).iloc[0]
        condition = f"i<{best_thr['i_limit']:g}, n_members>={int(best_thr['n_members_threshold'])}"
        lines.append(
            f"| {int(field_id):02d} | i<{best_base['i_limit']:g} | {fmt_pct(best_base['match_rate'])} | "
            f"{fmt_pct(best_base['purity_proxy'])} | `{condition}` | {fmt_int(best_thr['candidates_kept'])} | "
            f"{fmt_int(best_thr['matched_true_clusters'])} | {fmt_pct(best_thr['match_rate'])} | "
            f"{fmt_pct(best_thr['purity_proxy'])} | {fmt_float(best_thr['f1'], 3)} |"
        )
    return lines


def table_threshold_i22(thresholds: pd.DataFrame) -> list[str]:
    lines = [
        "| field | n_members >= | candidates kept | matched true clusters | match rate | purity proxy | candidate density |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sub = thresholds[np.isclose(thresholds["i_limit"], 22.0)].copy()
    for _, row in sub.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | {int(row['n_members_threshold'])} | {fmt_int(row['candidates_kept'])} | "
            f"{fmt_int(row['matched_true_clusters'])} | {fmt_pct(row['match_rate'])} | "
            f"{fmt_pct(row['purity_proxy'])} | {fmt_float(row['candidate_density'], 2)} |"
        )
    return lines


def write_md(summary: pd.DataFrame, thresholds: pd.DataFrame, coverage_summary: pd.DataFrame) -> None:
    lines = [
        "# Full 7band Fields 01-05 i-cut Blind-search Crossmatch Grid Tight Coverage",
        "",
        "## 覆盖定义修正",
        "",
        "这版不再用凸包作为 covered true clusters 的边界。",
        "每个 field 先用 full 7band 星系样本生成保守实际覆盖网格，然后所有 i-band cut 都使用同一个 field-level true-cluster 分母。",
        "",
        "- 覆盖网格：投影 RA-Dec 平面 `0.025 deg` cell。",
        "- 占据条件：每个 cell 至少 `3` 个 full 7band 星系。",
        "- 边界膨胀：`1` 个 cell，用来避免过度切掉边缘真实覆盖。",
        f"- 匹配半径：`{MATCH_RADIUS_MPC_H:.1f} pMpc/h`。",
        "- 红移条件：`|z_peak - z_cluster| <= 0.05 * (1 + z_cluster)`。",
        "- 统计采用一对多规则：只要 candidate 与 true cluster 满足空间和红移条件，就保留为有效 pair。",
        "",
        "## 输出文件",
        "",
        f"- 结果目录：`{OUTPUT_ROOT}`",
        f"- tight coverage true-cluster 数：`{OUTPUT_ROOT / 'tight_coverage_true_cluster_counts.csv'}`",
        f"- 基础 crossmatch 汇总：`{OUTPUT_ROOT / 'all_fields_i_cut_crossmatch_summary_tightcoverage.csv'}`",
        f"- `n_members` 阈值汇总：`{OUTPUT_ROOT / 'all_fields_i_cut_nmembers_crossmatch_summary_tightcoverage.csv'}`",
        f"- 汇总图：`{OUTPUT_ROOT / 'all_fields_i_cut_crossmatch_metrics_tightcoverage.png'}`",
        "",
        "![all fields tight coverage metrics](/Users/dengcanze/Documents/CSST/Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage/all_fields_i_cut_crossmatch_metrics_tightcoverage.png)",
        "",
        "## 覆盖区域修正前后对比",
        "",
        *table_coverage(coverage_summary),
        "",
        "## 基础 crossmatch 结果",
        "",
        *table_baseline(summary),
        "",
        "## 每个 field 的较优条件",
        "",
        "这里仍按 `F1 = 2 * match_rate * purity / (match_rate + purity)` 给出一个 recovery/purity 折中参考。",
        "",
        *table_best_by_field(summary, thresholds),
        "",
        "## `i<22.0` 下的 n_members 阈值扫描",
        "",
        *table_threshold_i22(thresholds),
        "",
        "## 简要分析",
        "",
    ]
    for field_id in sorted(summary["field_id"].unique()):
        sub = summary[summary["field_id"] == field_id].copy()
        best_match = sub.loc[sub["match_rate"].idxmax()]
        best_purity = sub.loc[sub["purity_proxy"].idxmax()]
        lines.append(
            f"- Field {int(field_id):02d}: 修正 coverage 后，基础 i-cut 最高 recovery 出现在 `i<{best_match['i_limit']:g}`，"
            f"match rate 为 `{best_match['match_rate']:.2%}`；最高 purity proxy 出现在 `i<{best_purity['i_limit']:g}`，"
            f"purity proxy 为 `{best_purity['purity_proxy']:.2%}`。"
        )
    lines.extend(
        [
            "",
            "修正后 field01/03 的 covered true clusters 分母显著下降，因此 recovery 明显上升；field04/05 因为原本覆盖更规整，变化较小。",
            "这个版本更适合作为后续 field-to-field recovery/purity 比较的基准。",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary, thresholds, coverage_summary = run_crossmatch_grid()
    plot_metric_grid(summary, thresholds)
    write_md(summary, thresholds, coverage_summary)
    print(f"Wrote summary: {OUTPUT_ROOT / 'all_fields_i_cut_crossmatch_summary_tightcoverage.csv'}")
    print(f"Wrote threshold summary: {OUTPUT_ROOT / 'all_fields_i_cut_nmembers_crossmatch_summary_tightcoverage.csv'}")
    print(f"Wrote markdown: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
