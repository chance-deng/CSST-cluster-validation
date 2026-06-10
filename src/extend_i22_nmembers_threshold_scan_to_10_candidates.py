#!/usr/bin/env python3
"""Extend i<22 n_members scans until each field keeps about ten candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
TIGHT_ROOT = PROJECT_ROOT / "Codex/result/full7band_i_cut_grid_crossmatch_all_fields_r1p5_tightcoverage"
OUT_DIR = TIGHT_ROOT / "extended_i22_nmembers_scan_to_10"
OUT_MD = PROJECT_ROOT / "Codex/notebook/full7band_i22_nmembers_extended_scan_to_10.md"


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


def fmt_int(value: float | int) -> str:
    return "-" if pd.isna(value) else f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    return "-" if pd.isna(value) else f"{float(value):.{digits}f}"


def load_field_paths(field_id: int) -> tuple[Path, Path, Path]:
    sub = TIGHT_ROOT / f"field{field_id:02d}_i_lt_22p0"
    match_csv = sub / f"field{field_id:02d}_crossmatch_matches.csv"
    covered_csv = sub / f"field{field_id:02d}_true_clusters_covered_tight.csv"
    summary = pd.read_csv(TIGHT_ROOT / "all_fields_i_cut_crossmatch_summary_tightcoverage.csv")
    row = summary[(summary["field_id"] == field_id) & np.isclose(summary["i_limit"], 22.0)].iloc[0]
    candidate_csv = Path(row["candidate_csv"])
    return candidate_csv, match_csv, covered_csv


def evaluate_threshold(candidates: pd.DataFrame, pairs: pd.DataFrame, total_clusters: int, threshold: int) -> dict:
    kept = candidates[candidates["n_members"] >= threshold].copy()
    kept_ids = set(kept["ID"].astype(int).tolist())
    use_pairs = pairs[pairs["candidate_id"].isin(kept_ids)].copy() if len(pairs) else pairs.copy()
    matched_clusters = int(use_pairs["cluster_index"].nunique()) if len(use_pairs) else 0
    participating = int(use_pairs["candidate_id"].nunique()) if len(use_pairs) else 0
    return {
        "n_members_threshold": int(threshold),
        "candidates_kept": int(len(kept)),
        "matched_true_clusters": matched_clusters,
        "match_rate": matched_clusters / total_clusters if total_clusters else np.nan,
        "participating_candidates": participating,
        "purity_proxy": participating / len(kept) if len(kept) else np.nan,
        "candidate_density": len(kept) / total_clusters if total_clusters else np.nan,
        "matched_pairs": int(len(use_pairs)),
        "median_distance_mpc_h": float(use_pairs["distance_mpc_h"].median()) if len(use_pairs) else np.nan,
        "median_abs_dz": float(use_pairs["abs_dz"].median()) if len(use_pairs) else np.nan,
    }


def scan_field(field_id: int) -> pd.DataFrame:
    candidate_csv, match_csv, covered_csv = load_field_paths(field_id)
    candidates = pd.read_csv(candidate_csv)
    pairs = pd.read_csv(match_csv)
    covered = pd.read_csv(covered_csv)
    total_clusters = len(covered)
    max_n = int(np.nanmax(candidates["n_members"].to_numpy(float)))
    rows = []
    stop_seen = False
    for threshold in range(1, max_n + 1):
        row = evaluate_threshold(candidates, pairs, total_clusters, threshold)
        row.update(
            {
                "field_id": field_id,
                "i_limit": 22.0,
                "total_covered_true_clusters": total_clusters,
                "total_candidates_at_i22": len(candidates),
                "candidate_csv": str(candidate_csv),
                "match_csv": str(match_csv),
            }
        )
        rows.append(row)
        if row["candidates_kept"] <= 10:
            stop_seen = True
            break
    if not stop_seen:
        # Keep the max threshold row even when the last row is still >10.
        pass
    return pd.DataFrame(rows)


def plot_field_scan(df: pd.DataFrame, out_png: Path) -> None:
    field_id = int(df["field_id"].iloc[0])
    fig, ax1 = plt.subplots(figsize=(7.4, 4.9), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(df["n_members_threshold"], df["match_rate"], color="#3C5488", lw=2.0, label="Match rate")
    ax1.plot(df["n_members_threshold"], df["purity_proxy"], color="#E64B35", lw=2.0, label="Purity proxy")
    ax2.plot(df["n_members_threshold"], df["candidates_kept"], color="#00A087", lw=1.8, ls="--", label="Candidates kept")
    ax2.axhline(10, color="#4D4D4D", lw=1.0, ls=":", label="10 candidates")
    ax1.set_xlabel(r"$n_{\rm members}$ threshold")
    ax1.set_ylabel("Fraction")
    ax2.set_ylabel("Candidates kept")
    ax1.set_ylim(0, 1.02)
    ax2.set_yscale("log")
    ax1.grid(True, ls="--", alpha=0.28)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=True, fontsize=9, loc="center right")
    ax1.set_title(f"Field {field_id:02d}, i<22.0 extended n_members scan", fontsize=14.5)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_md(all_df: pd.DataFrame, stop_summary: pd.DataFrame) -> None:
    lines = [
        "# Full 7band i<22.0 Extended n_members Threshold Scan",
        "",
        "## 定义",
        "",
        "- 使用 tight-coverage crossmatch 结果。",
        "- 只针对 `i<22.0` 的 blind-search candidate 表。",
        "- 阈值按整数 `n_members >= threshold` 扫描。",
        "- 每个 field 扫描到候选体数量 `<=10` 为止；若最后一个阈值正好跳过 10，则同时记录最接近且 `>=10` 的停止前状态。",
        "",
        f"- 结果目录：`{OUT_DIR}`",
        f"- 完整逐阈值表：`{OUT_DIR / 'i22_nmembers_extended_scan_all_fields.csv'}`",
        f"- 停止点汇总：`{OUT_DIR / 'i22_nmembers_extended_scan_stop_summary.csv'}`",
        "",
        "## 停止点汇总",
        "",
        "| field | stop type | n_members >= | candidates kept | matched true clusters | match rate | purity proxy | candidate density |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in stop_summary.iterrows():
        lines.append(
            f"| {int(row['field_id']):02d} | {row['stop_type']} | {int(row['n_members_threshold'])} | "
            f"{fmt_int(row['candidates_kept'])} | {fmt_int(row['matched_true_clusters'])} | "
            f"{fmt_pct(row['match_rate'])} | {fmt_pct(row['purity_proxy'])} | {fmt_float(row['candidate_density'], 3)} |"
        )
    lines.append("")
    lines.append("## 图")
    lines.append("")
    for field_id in sorted(all_df["field_id"].unique()):
        lines.extend(
            [
                f"### Field {int(field_id):02d}",
                "",
                f"![field{int(field_id):02d} extended scan]({OUT_DIR / f'field{int(field_id):02d}_i22_nmembers_extended_scan.png'})",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for field_id in [1, 2, 3, 4, 5]:
        df = scan_field(field_id)
        all_rows.append(df)
        out_png = OUT_DIR / f"field{field_id:02d}_i22_nmembers_extended_scan.png"
        plot_field_scan(df, out_png)
        print(f"field {field_id:02d}: thresholds={len(df)}, last_candidates={int(df.iloc[-1]['candidates_kept'])}")
    all_df = pd.concat(all_rows, ignore_index=True)
    all_csv = OUT_DIR / "i22_nmembers_extended_scan_all_fields.csv"
    all_df.to_csv(all_csv, index=False)

    stop_rows = []
    for field_id, sub in all_df.groupby("field_id"):
        before = sub[sub["candidates_kept"] >= 10]
        if len(before):
            row = before.iloc[-1].to_dict()
            row["stop_type"] = "last >=10"
            stop_rows.append(row)
        after = sub[sub["candidates_kept"] <= 10]
        if len(after):
            row = after.iloc[0].to_dict()
            row["stop_type"] = "first <=10"
            stop_rows.append(row)
    stop_summary = pd.DataFrame(stop_rows)
    stop_summary = stop_summary.drop_duplicates(
        subset=["field_id", "n_members_threshold", "candidates_kept", "matched_true_clusters"],
        keep="first",
    ).reset_index(drop=True)
    stop_csv = OUT_DIR / "i22_nmembers_extended_scan_stop_summary.csv"
    stop_summary.to_csv(stop_csv, index=False)
    write_md(all_df, stop_summary)
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {stop_csv}")
    print(f"Wrote: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
