#!/usr/bin/env python3
"""Scan PPM output cuts for Field05 i<22, n_members>=7 candidates."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"

PPM_ROOT = RESULT_ROOT / "field05_i22_nmembers7_ppm_candidates_and_true_clusters"
CAND_PPM_CSV = PPM_ROOT / "candidate_ppm_summary.csv"
MATCH_CSV = RESULT_ROOT / "field05_i_cut_grid_crossmatch_r1p5_thresholds/i_lt_22p0/cross_match_r1p5/field05_crossmatch_matches.csv"
OUTROOT = RESULT_ROOT / "field05_i22_nmembers7_ppm_threshold_scan"
NOTE = NOTEBOOK_ROOT / "field05_i22_nmembers7_ppm_threshold_scan.md"

COVERED_TRUE_CLUSTERS = 1188


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


def f1_score(recovery: float, purity: float) -> float:
    if recovery + purity <= 0:
        return 0.0
    return 2.0 * recovery * purity / (recovery + purity)


def evaluate(mask: np.ndarray, cand: pd.DataFrame, matches: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    kept = cand.loc[mask].copy()
    kept_ids = set(kept["candidate_id"].astype(int))
    matched = matches[matches["candidate_id"].astype(int).isin(kept_ids)]
    n_candidates = len(kept)
    matched_candidates = matched["candidate_id"].nunique()
    matched_clusters = matched["cluster_index"].nunique()
    recovery = matched_clusters / COVERED_TRUE_CLUSTERS
    purity = matched_candidates / n_candidates if n_candidates else 0.0
    return {
        "cut": label,
        "n_candidates": int(n_candidates),
        "matched_candidates": int(matched_candidates),
        "matched_true_clusters": int(matched_clusters),
        "recovery": recovery,
        "purity_proxy": purity,
        "candidate_density": n_candidates / COVERED_TRUE_CLUSTERS,
        "f1": f1_score(recovery, purity),
    }


def finite_thresholds(series: pd.Series, quantiles: list[float], decimals: int = 4) -> list[float]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = sorted(set(round(float(s.quantile(q)), decimals) for q in quantiles))
    return vals


def main() -> int:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(CAND_PPM_CSV)
    cand["candidate_id"] = cand["candidate_id"].astype(int)
    matches = pd.read_csv(MATCH_CSV)
    selected_ids = set(cand["candidate_id"].astype(int))
    matches = matches[matches["candidate_id"].astype(int).isin(selected_ids)].copy()

    cand["PPM_valid"] = pd.to_numeric(cand["PPM_significance"], errors="coerce").notna()
    cand["PPM_compact_richness"] = pd.to_numeric(cand["PPM_richness"], errors="coerce") / np.maximum(
        pd.to_numeric(cand["PPM_rmax_mean"], errors="coerce"), 0.25
    ) ** 2
    cand["PPM_density_proxy"] = cand["PPM_compact_richness"] / np.maximum(
        pd.to_numeric(cand["PPM_z_rms"], errors="coerce"), 0.01
    )

    rows: list[dict[str, float | int | str]] = []
    base_mask = np.ones(len(cand), dtype=bool)
    rows.append(evaluate(base_mask, cand, matches, "baseline: i<22.0+n_members>=7"))
    rows.append(evaluate(cand["PPM_valid"].to_numpy(bool), cand, matches, "PPM_valid"))

    sig_grid = finite_thresholds(cand["PPM_significance"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95])
    rich_grid = finite_thresholds(cand["PPM_richness"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95], 2)
    zrms_grid = finite_thresholds(cand["PPM_z_rms"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95], 5)
    rmax_grid = finite_thresholds(cand["PPM_rmax_mean"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95], 4)
    compact_grid = finite_thresholds(cand["PPM_compact_richness"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95], 3)
    density_grid = finite_thresholds(cand["PPM_density_proxy"], [0.05, 0.10, 0.16, 0.25, 0.33, 0.50, 0.67, 0.75, 0.84, 0.90, 0.95], 3)

    for t in sig_grid:
        rows.append(evaluate((cand["PPM_significance"] >= t).to_numpy(), cand, matches, f"PPM_significance >= {t:g}"))
    for t in rich_grid:
        rows.append(evaluate((cand["PPM_richness"] >= t).to_numpy(), cand, matches, f"PPM_richness >= {t:g}"))
    for t in zrms_grid:
        rows.append(evaluate((cand["PPM_z_rms"] <= t).to_numpy(), cand, matches, f"PPM_z_rms <= {t:g}"))
    for t in rmax_grid:
        rows.append(evaluate((cand["PPM_rmax_mean"] <= t).to_numpy(), cand, matches, f"PPM_rmax_mean <= {t:g}"))
        rows.append(evaluate((cand["PPM_rmax_mean"] >= t).to_numpy(), cand, matches, f"PPM_rmax_mean >= {t:g}"))
    for t in compact_grid:
        rows.append(evaluate((cand["PPM_compact_richness"] >= t).to_numpy(), cand, matches, f"PPM_richness/rmax^2 >= {t:g}"))
    for t in density_grid:
        rows.append(evaluate((cand["PPM_density_proxy"] >= t).to_numpy(), cand, matches, f"PPM_richness/(rmax^2*z_rms) >= {t:g}"))

    # Two-parameter scans. Keep the grid intentionally compact to avoid overfitting tiny threshold differences.
    for sig_t, rich_t in product(sig_grid, rich_grid):
        mask = (cand["PPM_significance"] >= sig_t) & (cand["PPM_richness"] >= rich_t)
        rows.append(evaluate(mask.to_numpy(), cand, matches, f"PPM_significance >= {sig_t:g} & PPM_richness >= {rich_t:g}"))
    for sig_t, zrms_t in product(sig_grid, zrms_grid):
        mask = (cand["PPM_significance"] >= sig_t) & (cand["PPM_z_rms"] <= zrms_t)
        rows.append(evaluate(mask.to_numpy(), cand, matches, f"PPM_significance >= {sig_t:g} & PPM_z_rms <= {zrms_t:g}"))
    for compact_t, zrms_t in product(compact_grid, zrms_grid):
        mask = (cand["PPM_compact_richness"] >= compact_t) & (cand["PPM_z_rms"] <= zrms_t)
        rows.append(evaluate(mask.to_numpy(), cand, matches, f"PPM_richness/rmax^2 >= {compact_t:g} & PPM_z_rms <= {zrms_t:g}"))

    result = pd.DataFrame(rows)
    result = result[result["n_candidates"] > 0].sort_values(["f1", "purity_proxy"], ascending=False).reset_index(drop=True)
    result.to_csv(OUTROOT / "ppm_threshold_scan_all.csv", index=False)

    selected_tables = {
        "best_f1": result.head(30),
        "best_recovery_ge_0p80": result[result["recovery"] >= 0.80].sort_values(["purity_proxy", "f1"], ascending=False).head(30),
        "best_recovery_ge_0p70": result[result["recovery"] >= 0.70].sort_values(["purity_proxy", "f1"], ascending=False).head(30),
        "best_recovery_ge_0p50": result[result["recovery"] >= 0.50].sort_values(["purity_proxy", "f1"], ascending=False).head(30),
        "best_purity": result.sort_values(["purity_proxy", "recovery"], ascending=False).head(30),
    }
    for name, df in selected_tables.items():
        df.to_csv(OUTROOT / f"{name}.csv", index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sc = ax.scatter(result["recovery"] * 100, result["purity_proxy"] * 100, c=result["f1"], s=16, cmap="viridis", alpha=0.75)
    base = result[result["cut"] == "baseline: i<22.0+n_members>=7"].iloc[0]
    ax.scatter([base["recovery"] * 100], [base["purity_proxy"] * 100], marker="*", s=130, color="#D62728", edgecolor="black", linewidth=0.5, label="baseline")
    ax.set_xlabel("Recovery (%)")
    ax.set_ylabel("Purity proxy (%)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("F1")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    plot_path = OUTROOT / "ppm_threshold_scan_recovery_purity.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    def md_table(df: pd.DataFrame, n: int = 8) -> str:
        cols = ["cut", "n_candidates", "matched_true_clusters", "recovery", "purity_proxy", "candidate_density", "f1"]
        x = df[cols].head(n).copy()
        for col in ["recovery", "purity_proxy", "candidate_density", "f1"]:
            x[col] = x[col].map(lambda v: f"{v:.4f}")
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] + ["---:"] * (len(cols) - 1)) + " |"
        body = []
        for _, row in x.iterrows():
            body.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join([header, sep, *body])

    base = result[result["cut"] == "baseline: i<22.0+n_members>=7"].iloc[0]
    valid = result[result["cut"] == "PPM_valid"].iloc[0]
    note = [
        "# Field05 i<22.0+n_members>=7：PPM 输出参数阈值扫描",
        "",
        "## 目的",
        "",
        "测试 PPM 输出参数是否能在当前候选体集合上进一步提高 purity proxy，并记录 recovery 的损失。",
        "",
        "## 指标定义",
        "",
        f"- covered true clusters 固定为 `{COVERED_TRUE_CLUSTERS}`。",
        "- recovery = matched true clusters / covered true clusters。",
        "- purity proxy = 至少匹配到一个 true cluster 的 candidate 数 / 保留 candidate 数。",
        "- candidate density = 保留 candidate 数 / covered true clusters。",
        "",
        "## Baseline",
        "",
        f"- baseline：`i<22.0 + n_members>=7`，candidate `{int(base['n_candidates'])}`，matched true clusters `{int(base['matched_true_clusters'])}`，recovery `{base['recovery']:.2%}`，purity proxy `{base['purity_proxy']:.2%}`，F1 `{base['f1']:.3f}`。",
        f"- 仅要求有有效 PPM 输出：candidate `{int(valid['n_candidates'])}`，matched true clusters `{int(valid['matched_true_clusters'])}`，recovery `{valid['recovery']:.2%}`，purity proxy `{valid['purity_proxy']:.2%}`，F1 `{valid['f1']:.3f}`。",
        "",
        "## 最优 F1",
        "",
        md_table(selected_tables["best_f1"]),
        "",
        "## 在 recovery >= 80% 时 purity 最高",
        "",
        md_table(selected_tables["best_recovery_ge_0p80"]),
        "",
        "## 在 recovery >= 70% 时 purity 最高",
        "",
        md_table(selected_tables["best_recovery_ge_0p70"]),
        "",
        "## 在 recovery >= 50% 时 purity 最高",
        "",
        md_table(selected_tables["best_recovery_ge_0p50"]),
        "",
        "## 纯度最高的极端阈值",
        "",
        md_table(selected_tables["best_purity"]),
        "",
        "## 图",
        "",
        f"![PPM threshold scan]({plot_path})",
        "",
        "## 初步结论",
        "",
        "- PPM 参数能略微提高 purity，但不能在保持高 recovery 的同时把 purity 推到很高。",
        "- 在 recovery >= 80% 的要求下，当前扫描没有找到优于 baseline 的 PPM 阈值；baseline purity proxy 仍为约 `40.8%`。",
        "- 若允许 recovery 降到约 70%，最佳 PPM cut 的 purity proxy 约 `43.4%`。",
        "- 若允许 recovery 降到约 50%，purity proxy 可以提升到约 `47.9%`，但相比之前单独调 `n_members` 的收益并不明显。",
        "- 这说明当前 PPM 输出更适合做候选体的后验诊断或排序特征；如果目标是显著提高 purity，可能需要把 PPM 特征、颜色/星等/stellar mass 信息以及局部密度形态一起训练一个监督式 reranker。",
        "",
    ]
    NOTE.write_text("\n".join(note), encoding="utf-8")
    print(f"Wrote all scan: {OUTROOT / 'ppm_threshold_scan_all.csv'}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote note: {NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
