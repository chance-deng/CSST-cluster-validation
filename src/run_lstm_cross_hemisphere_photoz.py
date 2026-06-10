#!/usr/bin/env python3
"""Cross-hemisphere CSST LSTM photo-z experiment.

Two directions are run:

1. Train on a 1:1 split of hemisphere_A and predict all selected sources in hemisphere_B.
2. Train on a 1:1 split of hemisphere_B and predict all selected sources in hemisphere_A.

The model architecture and feature construction reuse the existing Luo+2024
PyTorch implementation in ``train_csst_lstm_photoz_luo2024_torch_optimized``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from astropy.table import Table

from train_csst_lstm_photoz_luo2024_torch_optimized import (
    BANDS,
    build_model,
    choose_device,
    compute_train_weights,
    make_features,
    metrics,
    plot_ztrue_zphot,
    safe_num,
)


PROJECT_ROOT = Path("/Users/dengcanze/Documents/CSST")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Codex/result/lstm_cross_hemisphere_photoz/inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Codex/result/lstm_cross_hemisphere_photoz/cross_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train A->B and B->A CSST LSTM photo-z models.")
    p.add_argument("--hemisphere-a-csv", type=Path, default=DEFAULT_INPUT_DIR / "csst_photometry_hemisphere_A.csv")
    p.add_argument("--hemisphere-b-csv", type=Path, default=DEFAULT_INPUT_DIR / "csst_photometry_hemisphere_B.csv")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--chunk-size", type=int, default=500_000)
    p.add_argument("--test-fraction", type=float, default=0.5, help="1:1 split inside the training hemisphere.")
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--snr-threshold", type=float, default=10.0)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=2.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--hidden-units", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--mc-dropout-passes", type=int, default=5)
    p.add_argument("--zconf-half-width", type=float, default=0.05)
    p.add_argument("--use-colors", action="store_true", default=True)
    p.add_argument("--balanced-z-loss", action="store_true", default=True)
    p.add_argument("--z-bin-width", type=float, default=0.1)
    p.add_argument("--focus-z-min", type=float, default=1.0)
    p.add_argument("--focus-z-max", type=float, default=1.25)
    p.add_argument("--focus-weight", type=float, default=1.8)
    p.add_argument("--weight-clip", type=float, default=8.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--max-train-samples", type=int, default=None, help="Debug cap after selection for each training half.")
    p.add_argument("--max-target-samples", type=int, default=None, help="Debug cap after selection for each target half.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def selected_columns() -> list[str]:
    cols = ["row_idx", "id", "ra", "dec", "redshift", "sky_half", "field_id"]
    for band in BANDS:
        cols += [f"f_{band}", f"e_{band}"]
    return cols


def load_selected_rows(path: Path, args: argparse.Namespace, max_samples: int | None = None) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    row_idx: list[np.ndarray] = []
    ra_all: list[np.ndarray] = []
    dec_all: list[np.ndarray] = []
    sky_half: list[np.ndarray] = []
    field_id: list[np.ndarray] = []

    scanned = kept = 0
    for ci, df in enumerate(pd.read_csv(path, usecols=selected_columns(), chunksize=args.chunk_size), start=1):
        z = safe_num(df, "redshift")
        fluxes = np.stack([safe_num(df, f"f_{b}") for b in BANDS], axis=1)
        errors = np.stack([safe_num(df, f"e_{b}") for b in BANDS], axis=1)

        valid_flux = np.isfinite(fluxes) & np.isfinite(errors) & (fluxes > -90) & (errors > 0)
        g_idx = BANDS.index("g")
        i_idx = BANDS.index("i")
        snr_g = np.where(valid_flux[:, g_idx], fluxes[:, g_idx] / errors[:, g_idx], -np.inf)
        snr_i = np.where(valid_flux[:, i_idx], fluxes[:, i_idx] / errors[:, i_idx], -np.inf)
        mask = (
            np.isfinite(z)
            & (z > args.z_min)
            & (z <= args.z_max)
            & valid_flux.all(axis=1)
            & ((snr_g > args.snr_threshold) | (snr_i > args.snr_threshold))
        )

        scanned += len(df)
        if not np.any(mask):
            print(f"{path.name} chunk {ci}: scanned={scanned:,}, kept={kept:,}", flush=True)
            continue

        f = np.log1p(np.clip(fluxes[mask].astype(np.float32), 0.0, None))
        e = np.log1p(np.clip(errors[mask].astype(np.float32), 1e-12, None))
        xs.append(make_features(f, e, args.use_colors))
        ys.append(z[mask].astype(np.float32))
        ids.append(df.loc[mask, "id"].astype(str).to_numpy())
        row_idx.append(pd.to_numeric(df.loc[mask, "row_idx"], errors="coerce").to_numpy(np.int64))
        ra_all.append(pd.to_numeric(df.loc[mask, "ra"], errors="coerce").to_numpy(np.float32))
        dec_all.append(pd.to_numeric(df.loc[mask, "dec"], errors="coerce").to_numpy(np.float32))
        sky_half.append(df.loc[mask, "sky_half"].astype(str).to_numpy())
        field_id.append(pd.to_numeric(df.loc[mask, "field_id"], errors="coerce").to_numpy(np.int16))
        kept += int(mask.sum())
        print(f"{path.name} chunk {ci}: scanned={scanned:,}, kept={kept:,}", flush=True)

    data = {
        "x": np.concatenate(xs, axis=0),
        "z_true": np.concatenate(ys, axis=0),
        "id": np.concatenate(ids, axis=0),
        "row_idx": np.concatenate(row_idx, axis=0),
        "ra": np.concatenate(ra_all, axis=0),
        "dec": np.concatenate(dec_all, axis=0),
        "sky_half": np.concatenate(sky_half, axis=0),
        "field_id": np.concatenate(field_id, axis=0),
    }

    if max_samples is not None and len(data["z_true"]) > max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(data["z_true"]), size=max_samples, replace=False)
        data = {key: value[idx] for key, value in data.items()}
    return data


def fit_scaler(x_train: np.ndarray) -> dict[str, list[float]]:
    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0).astype(np.float32)
    sigma = x_train.reshape(-1, x_train.shape[-1]).std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0
    return {"mean": mu.tolist(), "std": sigma.tolist(), "features": ["log1p_flux", "log1p_flux_err", "adjacent_log_color"]}


def standardize(x: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    mu = np.asarray(scaler["mean"], dtype=np.float32)
    sigma = np.asarray(scaler["std"], dtype=np.float32)
    return ((x - mu.reshape(1, 1, -1)) / sigma.reshape(1, 1, -1)).astype(np.float32)


def predict_batched(model, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            chunks.append(model(xb).detach().cpu().numpy().astype(np.float32))
            del xb
    return np.concatenate(chunks)


def predict_mc_stats(model, x: np.ndarray, args: argparse.Namespace, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if args.mc_dropout_passes <= 0:
        pred = predict_batched(model, x, args.batch_size, device)
        return pred, pred, np.full(len(pred), np.nan, dtype=np.float32), np.full(len(pred), np.nan, dtype=np.float32)

    model.train()
    samples = []
    with torch.inference_mode():
        for k in range(args.mc_dropout_passes):
            chunks = []
            for start in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[start:start + args.batch_size]).to(device)
                chunks.append(model(xb).detach().cpu().numpy().astype(np.float32))
                del xb
            samples.append(np.concatenate(chunks))
            print(f"MC dropout pass {k + 1}/{args.mc_dropout_passes}", flush=True)
    arr = np.stack(samples, axis=0)
    z_pdf_peak = np.median(arr, axis=0).astype(np.float32)
    z_mc_mean = arr.mean(axis=0).astype(np.float32)
    z_mc_std = arr.std(axis=0).astype(np.float32)
    width = args.zconf_half_width * (1.0 + z_pdf_peak)
    zconf = np.mean(np.abs(arr - z_pdf_peak[None, :]) <= width[None, :], axis=0).astype(np.float32)
    return z_pdf_peak, z_mc_mean, z_mc_std, zconf


def train_model(direction: str, train_data: dict[str, np.ndarray], args: argparse.Namespace, out_dir: Path):
    import torch
    from torch import nn

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_data["z_true"]))
    n_holdout = int(round(len(perm) * args.test_fraction))
    holdout_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    x_fit_raw = train_data["x"][fit_idx]
    y_fit = train_data["z_true"][fit_idx]
    x_holdout_raw = train_data["x"][holdout_idx]
    y_holdout = train_data["z_true"][holdout_idx]

    scaler = fit_scaler(x_fit_raw)
    x_fit = standardize(x_fit_raw, scaler)
    x_holdout = standardize(x_holdout_raw, scaler)

    train_args = SimpleNamespace(**vars(args))
    train_args.focus_axis = "true"
    weights = compute_train_weights(y_fit, train_args)

    n_val = int(round(len(x_fit) * args.validation_fraction))
    split = len(x_fit) - n_val
    x_val, y_val = x_fit[split:], y_fit[split:]
    x_train, y_train, w_train = x_fit[:split], y_fit[:split], weights[:split]

    device = choose_device(args.device)
    print(f"{direction}: using device {device}", flush=True)
    model = build_model(train_args).to(device)
    loss_fn = nn.MSELoss(reduction="none")
    eval_loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        epoch_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size].cpu().numpy()
            xb = torch.from_numpy(x_train[batch]).to(device)
            yb = torch.from_numpy(y_train[batch]).to(device)
            wb = torch.from_numpy(w_train[batch]).to(device)
            pred = model(xb)
            loss = (loss_fn(pred, yb) * wb).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * len(batch)
            del xb, yb, wb, pred, loss
        epoch_loss /= max(len(x_train), 1)

        model.eval()
        total = 0.0
        n_seen = 0
        with torch.no_grad():
            for start in range(0, len(x_val), args.batch_size):
                xb = torch.from_numpy(x_val[start:start + args.batch_size]).to(device)
                yb = torch.from_numpy(y_val[start:start + args.batch_size]).to(device)
                loss = eval_loss_fn(model(xb), yb)
                total += float(loss.item()) * len(xb)
                n_seen += len(xb)
                del xb, yb, loss
        val_loss = total / max(n_seen, 1)
        print(f"{direction} epoch {epoch:03d}: train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"{direction}: early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_cross_hemisphere_lstm.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    (out_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    z_holdout = predict_batched(model, x_holdout, args.batch_size, device)
    holdout_stat = metrics(y_holdout, z_holdout)
    holdout = pd.DataFrame(
        {
            "row_idx": train_data["row_idx"][holdout_idx],
            "id": train_data["id"][holdout_idx],
            "ra": train_data["ra"][holdout_idx],
            "dec": train_data["dec"][holdout_idx],
            "z_true": y_holdout,
            "z_phot": z_holdout,
            "sky_half": train_data["sky_half"][holdout_idx],
            "field_id": train_data["field_id"][holdout_idx],
            "direction": direction,
            "sample_role": "internal_holdout",
        }
    )
    holdout.to_csv(out_dir / "internal_holdout_predictions.csv", index=False)
    pd.DataFrame([{**holdout_stat, "direction": direction, "sample_role": "internal_holdout"}]).to_csv(
        out_dir / "internal_holdout_metrics.csv",
        index=False,
    )
    plot_ztrue_zphot(y_holdout, z_holdout, holdout_stat, out_dir / "internal_holdout_ztrue_vs_zphot.png", f"{direction} internal holdout")
    return model, scaler, device, holdout_stat


def predict_target(direction: str, model, scaler, device, target_data: dict[str, np.ndarray], args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    x_target = standardize(target_data["x"], scaler)
    z_phot, z_mc_mean, z_mc_std, zconf = predict_mc_stats(model, x_target, args, device)
    stat = metrics(target_data["z_true"], z_phot)
    pred = pd.DataFrame(
        {
            "row_idx": target_data["row_idx"],
            "id": target_data["id"],
            "ra": target_data["ra"],
            "dec": target_data["dec"],
            "z_true": target_data["z_true"],
            "z_phot": z_phot,
            "z_mc_mean": z_mc_mean,
            "z_mc_std": z_mc_std,
            "zConf": zconf,
            "sky_half": target_data["sky_half"],
            "field_id": target_data["field_id"],
            "direction": direction,
            "sample_role": "cross_target",
        }
    )
    pred.to_csv(out_dir / "cross_target_predictions.csv", index=False)
    pd.DataFrame([{**stat, "direction": direction, "sample_role": "cross_target"}]).to_csv(out_dir / "cross_target_metrics.csv", index=False)
    plot_ztrue_zphot(target_data["z_true"], z_phot, stat, out_dir / "cross_target_ztrue_vs_zphot.png", f"{direction} cross target")
    return pred


def write_blindsearch_fits(pred: pd.DataFrame, out_fits: Path) -> None:
    zfinal = pd.to_numeric(pred["z_phot"], errors="coerce").to_numpy(np.float32)
    sigma = pd.to_numeric(pred["z_mc_std"], errors="coerce").to_numpy(np.float32)
    fallback = 0.03 * (1.0 + zfinal)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback).astype(np.float32)
    table = Table()
    table["ra"] = pd.to_numeric(pred["ra"], errors="coerce").to_numpy(np.float32)
    table["dec"] = pd.to_numeric(pred["dec"], errors="coerce").to_numpy(np.float32)
    table["zfinal"] = zfinal
    table["zpdf_l68"] = np.clip(zfinal - sigma, 0.0, 3.8).astype(np.float32)
    table["zpdf_u68"] = np.clip(zfinal + sigma, 0.0, 3.8).astype(np.float32)
    table.write(out_fits, overwrite=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hemisphere A photometry...", flush=True)
    hemi_a = load_selected_rows(args.hemisphere_a_csv, args, args.max_train_samples)
    print("Loading hemisphere B photometry...", flush=True)
    hemi_b = load_selected_rows(args.hemisphere_b_csv, args, args.max_train_samples)

    dir_ab = args.output_dir / "train_A_predict_B"
    model_ab, scaler_ab, device_ab, stat_ab_holdout = train_model("train_A_predict_B", hemi_a, args, dir_ab)
    pred_b = predict_target("train_A_predict_B", model_ab, scaler_ab, device_ab, hemi_b, args, dir_ab)

    dir_ba = args.output_dir / "train_B_predict_A"
    model_ba, scaler_ba, device_ba, stat_ba_holdout = train_model("train_B_predict_A", hemi_b, args, dir_ba)
    pred_a = predict_target("train_B_predict_A", model_ba, scaler_ba, device_ba, hemi_a, args, dir_ba)

    combined = pd.concat([pred_a, pred_b], ignore_index=True).sort_values("row_idx").reset_index(drop=True)
    combined_csv = args.output_dir / "cross_hemisphere_lstm_predictions_combined.csv"
    combined.to_csv(combined_csv, index=False)
    write_blindsearch_fits(combined, args.output_dir / "cross_hemisphere_lstm_photoz_blindsearch.fits")

    metric_rows = []
    for subdir in [dir_ab, dir_ba]:
        metric_rows.append(pd.read_csv(subdir / "internal_holdout_metrics.csv"))
        metric_rows.append(pd.read_csv(subdir / "cross_target_metrics.csv"))
    metrics_df = pd.concat(metric_rows, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "cross_hemisphere_lstm_metrics.csv", index=False)

    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    with (args.output_dir / "cross_hemisphere_lstm_summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# CSST Cross-Hemisphere LSTM Photo-z\n\n")
        fh.write("Two models are trained without cutting fields: train on hemisphere A and predict hemisphere B, then train on hemisphere B and predict hemisphere A.\n\n")
        fh.write(f"- Hemisphere A selected rows: `{len(hemi_a['z_true']):,}`\n")
        fh.write(f"- Hemisphere B selected rows: `{len(hemi_b['z_true']):,}`\n")
        fh.write(f"- Internal split in each training hemisphere: `1:1`, test fraction `{args.test_fraction}`\n")
        fh.write(f"- Combined prediction CSV: `{combined_csv}`\n")
        fh.write(f"- Blind-search FITS: `{args.output_dir / 'cross_hemisphere_lstm_photoz_blindsearch.fits'}`\n\n")
        fh.write("| direction | sample_role | N | sigma_NMAD | outlier_fraction | bias |\n")
        fh.write("|---|---|---:|---:|---:|---:|\n")
        for row in metrics_df.to_dict(orient="records"):
            fh.write(
                f"| {row['direction']} | {row['sample_role']} | {int(row['N']):,} | "
                f"{row['sigma_NMAD']:.6f} | {row['outlier_fraction']:.6f} | {row['bias']:.6f} |\n"
            )

    print(f"Wrote combined predictions: {combined_csv}")
    print(f"Wrote blind-search FITS: {args.output_dir / 'cross_hemisphere_lstm_photoz_blindsearch.fits'}")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
