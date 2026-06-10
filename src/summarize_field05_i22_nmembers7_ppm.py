#!/usr/bin/env python3
"""Summarize Field05 i<22, n_members>=7 PPM outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
RESULT_ROOT = PROJECT_ROOT / "Codex/result"
NOTEBOOK_ROOT = PROJECT_ROOT / "Codex/notebook"

OUTROOT = RESULT_ROOT / "field05_i22_nmembers7_ppm_candidates_and_true_clusters"
CAND_CSV = OUTROOT / "candidate_ppm_summary.csv"
TRUE_CSV = OUTROOT / "true_cluster_ppm_summary.csv"
NOTE = NOTEBOOK_ROOT / "field05_i22_nmembers7_ppm_candidates_and_true_clusters.md"

SEVENBAND_BG = RESULT_ROOT / "blindsearch_inputs/field05_7band_i_band_cuts/field05_7band_i_lt_22p0_blindsearch_input.fits"
TRUE_BG = RESULT_ROOT / "galaxies_C6_field05_blindsearch_input.fits"


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


def stats(df: pd.DataFrame, col: str) -> dict[str, float | int]:
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "valid": int(s.notna().sum()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p16": float(s.quantile(0.16)),
        "p84": float(s.quantile(0.84)),
        "max": float(s.max()),
    }


def make_summary_plot(cand: pd.DataFrame, true: pd.DataFrame) -> Path:
    setup_style()
    out = OUTROOT / "field05_i22_nmembers7_ppm_summary.png"

    cand_sig = pd.to_numeric(cand["PPM_significance"], errors="coerce").dropna()
    true_sig = pd.to_numeric(true["PPM_significance"], errors="coerce").dropna()
    cand_rich = pd.to_numeric(cand["PPM_richness"], errors="coerce").dropna()
    true_rich = pd.to_numeric(true["PPM_richness"], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    axes[0].hist(cand_sig, bins=np.linspace(2, 8, 36), histtype="step", lw=1.7, density=True, label="Candidates")
    axes[0].hist(true_sig, bins=np.linspace(2, 8, 36), histtype="step", lw=1.7, density=True, label="Matched true clusters")
    axes[0].set_xlabel("PPM significance")
    axes[0].set_ylabel("Normalized density")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].hist(np.log10(cand_rich.clip(lower=1)), bins=np.linspace(0, 4.2, 38), histtype="step", lw=1.7, density=True, label="Candidates")
    axes[1].hist(np.log10(true_rich.clip(lower=1)), bins=np.linspace(0, 4.2, 38), histtype="step", lw=1.7, density=True, label="Matched true clusters")
    axes[1].set_xlabel(r"$\log_{10}(\mathrm{PPM\ richness})$")
    axes[1].set_ylabel("Normalized density")
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def fmt_stat(d: dict[str, float | int]) -> str:
    return f"{d['median']:.3f} ({d['p16']:.3f}-{d['p84']:.3f})"


def main() -> int:
    cand = pd.read_csv(CAND_CSV)
    true = pd.read_csv(TRUE_CSV)
    plot = make_summary_plot(cand, true)

    cand_sig = stats(cand, "PPM_significance")
    true_sig = stats(true, "PPM_significance")
    cand_rich = stats(cand, "PPM_richness")
    true_rich = stats(true, "PPM_richness")
    cand_z = stats(cand, "PPM_z_mean")
    true_z = stats(true, "PPM_z_mean")

    lines = [
        "# Field05 v1.1 复分析：i<22.0 + n_members>=7 的 PPM 结果",
        "",
        "## 输入定义",
        "",
        "- 筛选条件：`i < 22.0` 且 blind-search `n_members >= 7`。",
        "- 该条件对应先前最优 F1 平衡结果：`candidates kept = 4,590`，`matched true clusters = 956`，`recovery = 80.47%`，`purity proxy = 40.78%`，`F1 = 0.541`。",
        f"- Candidate PPM 使用 7band 亮星样本作为背景：`{SEVENBAND_BG}`。",
        f"- True-cluster PPM 使用 C6 完整真实星表作为背景：`{TRUE_BG}`。",
        "- 两组 PPM 的背景面积分别由各自星表的实际 RA-Dec 覆盖凸包估计，不共用旧的固定面积。",
        "",
        "## 输出文件",
        "",
        f"- Candidate PPM summary：`{CAND_CSV}`",
        f"- True-cluster PPM summary：`{TRUE_CSV}`",
        f"- Candidate PPM 图目录：`{OUTROOT / 'plots/candidate'}`",
        f"- True-cluster PPM 图目录：`{OUTROOT / 'plots/true_cluster'}`",
        f"- Candidate PPM 文本目录：`{OUTROOT / 'ppm_outputs/candidate'}`",
        f"- True-cluster PPM 文本目录：`{OUTROOT / 'ppm_outputs/true_cluster'}`",
        "",
        "## 完整性检查",
        "",
        f"- Candidate summary 行数：`{len(cand)}`。",
        f"- True-cluster summary 行数：`{len(true)}`。",
        f"- Candidate 图数：`{count_files(OUTROOT / 'plots/candidate', '*.png')}`。",
        f"- True-cluster 图数：`{count_files(OUTROOT / 'plots/true_cluster', '*.png')}`。注：目录中包含前面 smoke test 留下的少量图，因此可能略多于 956。",
        f"- Candidate PPM 文本数：`{count_files(OUTROOT / 'ppm_outputs/candidate', '**/*.txt')}`。",
        f"- True-cluster PPM 文本数：`{count_files(OUTROOT / 'ppm_outputs/true_cluster', '**/*.txt')}`。注：同样包含 smoke test 产物。",
        "",
        "## PPM 统计",
        "",
        "| sample | N | valid PPM | PPM significance median (16-84%) | PPM richness median (16-84%) | PPM z_mean median (16-84%) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 7band candidates | {len(cand)} | {cand_sig['valid']} | {fmt_stat(cand_sig)} | {fmt_stat(cand_rich)} | {fmt_stat(cand_z)} |",
        f"| matched true clusters | {len(true)} | {true_sig['valid']} | {fmt_stat(true_sig)} | {fmt_stat(true_rich)} | {fmt_stat(true_z)} |",
        "",
        "![PPM summary](" + str(plot) + ")",
        "",
        "## 初步解读",
        "",
        "- Candidate PPM 中有 `4041/4590` 个目标得到有效 PPM 结构，说明 `i<22.0 + n_members>=7` 筛出的候选体大部分在 7band 亮星背景中仍有可识别的局部红移-角距离峰。",
        "- Matched true clusters 在 C6 完整星表背景下的 PPM significance 显著更高，中位数约 `7.84`，richness 中位数约 `1022`。这与真实星表包含完整暗晕成员、而 7band 亮星样本只是稀疏 tracer 的预期一致。",
        "- 两组 PPM 不应直接用 richness 绝对值比较物理富度，因为背景星表完全不同；更合理的用途是分别检查候选体在观测 tracer 中是否形成峰，以及 true cluster 在完整真实星表中是否确实对应高密度结构。",
        "",
    ]
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote plot: {plot}")
    print(f"Wrote note: {NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
