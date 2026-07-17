#!/usr/bin/env python3
"""Configurable blind overdensity search for photometric-redshift catalogs.

This module is a cleaned, reusable implementation of the CSST blind cluster
search used in the manuscript.  It detects overdensity peaks from sky
coordinates and photometric redshifts only; it does not use truth-cluster
labels, halo information, red-sequence priors, PPM validation, or cross-match
results during candidate detection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.ndimage import gaussian_filter, maximum_filter


@dataclass
class CatalogConfig:
    """Catalog input settings.

    EN: Store the input catalog path and column mapping used by the blind
    search.
    ZH: 保存输入星表路径和 blind search 使用的列名映射。
    """

    path: Path
    hdu: int | None = None
    ra_column: str = "ra"
    dec_column: str = "dec"
    redshift_column: str = "zfinal"
    magnitude_column: str | None = None
    magnitude_limit: float | None = None
    exclude_flag_star: bool = False
    exclude_flag_star_hsc: bool = False
    require_warn_flag_zero: bool = False


@dataclass
class CosmologyConfig:
    """Flat Lambda-CDM cosmology settings.

    EN: Define the cosmology used to convert physical smoothing and matching
    scales into angular units.
    ZH: 定义宇宙学参数，用于把物理平滑尺度和合并半径转换为角尺度。
    """

    h0: float = 67.66
    omega_m: float = 0.3111


@dataclass
class SliceConfig:
    """Redshift slicing settings.

    EN: Configure the overlapping redshift slices used to project the
    photometric-redshift catalog.
    ZH: 配置重叠红移切片，用于将 photo-z 星表投影到二维天空平面。
    """

    z_min: float = 0.0
    z_max: float = 2.2
    z_step: float = 0.01
    half_width_factor: float = 0.03
    min_galaxies_per_slice: int = 5


@dataclass
class MapConfig:
    """Density-map construction settings.

    EN: Configure the sky-grid pixel size, Gaussian smoothing scales, and local
    background kernel.
    ZH: 配置天空网格像素尺寸、高斯平滑物理尺度和局部背景核。
    """

    pixel_size_arcmin: float = 0.3
    smoothing_scales_mpc: list[float] = field(default_factory=lambda: [0.4, 0.8, 1.2])
    background_sigma_factor: float = 15.0
    sigma_pixel_min: float = 1.0
    sigma_pixel_max: float = 15.0
    background_floor: float = 1e-5
    maximum_filter_size: int = 5


@dataclass
class CandidateDensityConfig:
    """Aperture candidate-overdensity settings.

    EN: Configure the aperture overdensity proxy based on N_member and the
    global redshift-slice background surface density.
    ZH: 配置基于 N_member 和全局红移切片背景面密度的孔径过密度参考量。
    """

    background_area_deg2: float | None = None
    aperture_radius_factor: float = 1.5
    footprint_cell_deg: float | None = None


@dataclass
class PeakConfig:
    """Raw peak selection settings.

    EN: Configure the overdensity and significance thresholds used to retain
    raw slice-level peaks.
    ZH: 配置保留单切片 raw peaks 的过密度和显著性阈值。
    """

    significance_min: float = 0.2
    overdensity_min: float = 0.5
    refinement_radius_factor: float = 1.5


@dataclass
class MergeConfig:
    """Greedy non-maximum-suppression settings.

    EN: Configure the spatial-redshift duplicate-merging rule and persistence
    filter for final candidates.
    ZH: 配置空间-红移重复探测合并规则，以及最终候选体的持续性筛选。
    """

    spatial_radius_mpc_h: float = 1.0
    redshift_factor: float = 0.04
    min_detected_slices: int = 2
    z_bin: float = 0.02
    sky_cell_size_deg: float = 0.2


@dataclass
class OutputConfig:
    """Output-path settings.

    EN: Configure where candidate tables, raw peak tables, slice summaries, and
    run metadata are written.
    ZH: 配置候选体表、raw peak 表、切片统计和运行元数据的输出位置。
    """

    directory: Path
    candidate_filename: str = "blindsearch_candidates.csv"
    raw_peak_filename: str = "blindsearch_raw_peaks.csv"
    slice_filename: str = "blindsearch_slice_stats.csv"
    merge_filename: str = "blindsearch_merge_assignments.csv"
    summary_filename: str = "blindsearch_summary.json"


@dataclass
class BlindSearchConfig:
    """Top-level blind-search configuration.

    EN: Bundle all configuration sections needed by a complete blind-search
    run.
    ZH: 汇总一次完整 blind search 运行所需的全部配置项。
    """

    catalog: CatalogConfig
    cosmology: CosmologyConfig
    slices: SliceConfig
    density_map: MapConfig
    candidate_density: CandidateDensityConfig
    peaks: PeakConfig
    merge: MergeConfig
    output: OutputConfig


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file.

    EN: Read the user configuration. YAML is preferred when PyYAML is
    installed; JSON is also supported as a dependency-light fallback.
    ZH: 读取用户配置文件。若安装了 PyYAML，则优先支持 YAML；同时支持 JSON 作为轻量回退方案。
    """

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON or install pyyaml.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping/dictionary.")
    return data


def require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required configuration section.

    EN: Validate that a named configuration section exists and is a mapping.
    ZH: 检查指定配置块是否存在，并确认其类型为字典。
    """

    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {name}")
    return section


def build_config(data: dict[str, Any]) -> BlindSearchConfig:
    """Build typed configuration objects from a raw mapping.

    EN: Convert the nested dictionary read from YAML/JSON into dataclass
    objects with defaults for optional parameters.
    ZH: 将 YAML/JSON 读取出的嵌套字典转换为 dataclass 配置对象，并为可选参数补默认值。
    """

    catalog = require_section(data, "catalog")
    cosmology = data.get("cosmology", {})
    slices = data.get("slices", {})
    density_map = data.get("density_map", {})
    candidate_density = data.get("candidate_density", {})
    peaks = data.get("peaks", {})
    merge = data.get("merge", {})
    output = require_section(data, "output")

    return BlindSearchConfig(
        catalog=CatalogConfig(
            path=Path(catalog["path"]).expanduser(),
            hdu=(None if catalog.get("hdu") is None else int(catalog.get("hdu"))),
            ra_column=str(catalog.get("ra_column", "ra")),
            dec_column=str(catalog.get("dec_column", "dec")),
            redshift_column=str(catalog.get("redshift_column", "zfinal")),
            magnitude_column=catalog.get("magnitude_column"),
            magnitude_limit=catalog.get("magnitude_limit"),
            exclude_flag_star=bool(catalog.get("exclude_flag_star", False)),
            exclude_flag_star_hsc=bool(catalog.get("exclude_flag_star_hsc", False)),
            require_warn_flag_zero=bool(catalog.get("require_warn_flag_zero", False)),
        ),
        cosmology=CosmologyConfig(
            h0=float(cosmology.get("h0", 67.66)),
            omega_m=float(cosmology.get("omega_m", 0.3111)),
        ),
        slices=SliceConfig(
            z_min=float(slices.get("z_min", 0.0)),
            z_max=float(slices.get("z_max", 2.2)),
            z_step=float(slices.get("z_step", 0.01)),
            half_width_factor=float(slices.get("half_width_factor", 0.03)),
            min_galaxies_per_slice=int(slices.get("min_galaxies_per_slice", 5)),
        ),
        density_map=MapConfig(
            pixel_size_arcmin=float(density_map.get("pixel_size_arcmin", 0.3)),
            smoothing_scales_mpc=[float(x) for x in density_map.get("smoothing_scales_mpc", [0.4, 0.8, 1.2])],
            background_sigma_factor=float(density_map.get("background_sigma_factor", 15.0)),
            sigma_pixel_min=float(density_map.get("sigma_pixel_min", 1.0)),
            sigma_pixel_max=float(density_map.get("sigma_pixel_max", 15.0)),
            background_floor=float(density_map.get("background_floor", 1e-5)),
            maximum_filter_size=int(density_map.get("maximum_filter_size", 5)),
        ),
        candidate_density=CandidateDensityConfig(
            background_area_deg2=(
                None
                if candidate_density.get("background_area_deg2") is None
                else float(candidate_density.get("background_area_deg2"))
            ),
            aperture_radius_factor=float(candidate_density.get("aperture_radius_factor", 1.5)),
            footprint_cell_deg=(
                None
                if candidate_density.get("footprint_cell_deg") is None
                else float(candidate_density.get("footprint_cell_deg"))
            ),
        ),
        peaks=PeakConfig(
            significance_min=float(peaks.get("significance_min", 0.2)),
            overdensity_min=float(peaks.get("overdensity_min", 0.5)),
            refinement_radius_factor=float(peaks.get("refinement_radius_factor", 1.5)),
        ),
        merge=MergeConfig(
            spatial_radius_mpc_h=float(merge.get("spatial_radius_mpc_h", 1.0)),
            redshift_factor=float(merge.get("redshift_factor", 0.04)),
            min_detected_slices=int(merge.get("min_detected_slices", 2)),
            z_bin=float(merge.get("z_bin", 0.02)),
            sky_cell_size_deg=float(merge.get("sky_cell_size_deg", 0.2)),
        ),
        output=OutputConfig(
            directory=Path(output["directory"]).expanduser(),
            candidate_filename=str(output.get("candidate_filename", "blindsearch_candidates.csv")),
            raw_peak_filename=str(output.get("raw_peak_filename", "blindsearch_raw_peaks.csv")),
            slice_filename=str(output.get("slice_filename", "blindsearch_slice_stats.csv")),
            merge_filename=str(output.get("merge_filename", "blindsearch_merge_assignments.csv")),
            summary_filename=str(output.get("summary_filename", "blindsearch_summary.json")),
        ),
    )


def make_cosmology(config: CosmologyConfig) -> FlatLambdaCDM:
    """Create an Astropy cosmology object.

    EN: Instantiate the flat Lambda-CDM cosmology used by all angular-distance
    conversions.
    ZH: 创建用于所有角径距离转换的平直 Lambda-CDM 宇宙学对象。
    """

    return FlatLambdaCDM(H0=config.h0, Om0=config.omega_m)


def load_catalog(config: CatalogConfig) -> pd.DataFrame:
    """Load and clean the input galaxy catalog.

    EN: Read FITS/CSV/ECSV input, apply finite-value checks, optional magnitude
    cuts, and return standardized columns named ra, dec, and z.
    ZH: 读取 FITS/CSV/ECSV 输入，执行有限值检查和可选星等截断，并返回标准列 ra、dec、z。
    """

    if not config.path.exists():
        raise FileNotFoundError(f"Input catalog not found: {config.path}")

    suffix = config.path.suffix.lower()
    if suffix in {".fits", ".fit", ".fz"}:
        read_kwargs: dict[str, Any] = {"memmap": True}
        if config.hdu is not None:
            read_kwargs["hdu"] = config.hdu
        table = Table.read(config.path, **read_kwargs)
        needed_names = [
            config.ra_column,
            config.dec_column,
            config.redshift_column,
            "flag_star",
            "flag_star_hsc",
            "warn_flag",
        ]
        if config.magnitude_column:
            needed_names.append(config.magnitude_column)
        available_names = [name for name in needed_names if name in table.colnames]
        source = table[available_names].to_pandas()
    elif suffix in {".csv", ".txt"}:
        source = pd.read_csv(config.path)
    elif suffix in {".ecsv"}:
        source = Table.read(config.path, format="ascii.ecsv").to_pandas()
    else:
        table = Table.read(config.path)
        source = table.to_pandas()

    needed = [config.ra_column, config.dec_column, config.redshift_column]
    if config.magnitude_column:
        needed.append(config.magnitude_column)
    missing = [name for name in needed if name not in source.columns]
    if missing:
        raise ValueError(f"Missing required catalog columns: {missing}")

    df = pd.DataFrame(
        {
            "ra": np.asarray(source[config.ra_column], dtype=float),
            "dec": np.asarray(source[config.dec_column], dtype=float),
            "z": np.asarray(source[config.redshift_column], dtype=float),
        }
    )
    mask = np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["z"])
    if config.magnitude_column and config.magnitude_limit is not None:
        mag = np.asarray(source[config.magnitude_column], dtype=float)
        mask &= np.isfinite(mag) & (mag < float(config.magnitude_limit))
    if config.exclude_flag_star and "flag_star" in source.columns:
        mask &= ~np.asarray(source["flag_star"], dtype=bool)
    if config.exclude_flag_star_hsc and "flag_star_hsc" in source.columns:
        mask &= np.asarray(source["flag_star_hsc"], dtype=float) == 0
    if config.require_warn_flag_zero and "warn_flag" in source.columns:
        mask &= np.asarray(source["warn_flag"], dtype=float) == 0
    return df.loc[mask].reset_index(drop=True)


def restrict_redshift_range(catalog: pd.DataFrame, config: SliceConfig) -> pd.DataFrame:
    """Restrict the catalog to the configured redshift range.

    EN: Keep galaxies with photometric redshifts inside the inclusive global
    search interval.
    ZH: 保留 photo-z 位于全局搜索红移范围内的星系。
    """

    mask = (catalog["z"] >= config.z_min) & (catalog["z"] <= config.z_max)
    return catalog.loc[mask].reset_index(drop=True)


def build_sky_bins(catalog: pd.DataFrame, pixel_size_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Build RA and Dec bin edges for the sky grid.

    EN: Create regular sky-plane bins covering the input catalog footprint.
    ZH: 根据输入星表覆盖范围构建规则 RA-Dec 天空网格边界。
    """

    ra_min, ra_max = float(catalog["ra"].min()), float(catalog["ra"].max())
    dec_min, dec_max = float(catalog["dec"].min()), float(catalog["dec"].max())
    ra_bins = np.arange(ra_min, ra_max + pixel_size_deg, pixel_size_deg)
    dec_bins = np.arange(dec_min, dec_max + pixel_size_deg, pixel_size_deg)
    if len(ra_bins) < 2 or len(dec_bins) < 2:
        raise ValueError("Catalog footprint is too small for the configured pixel size.")
    return ra_bins, dec_bins


def build_footprint_grid(
    catalog: pd.DataFrame,
    cell_deg: float | None,
) -> dict[str, Any] | None:
    """Build an occupied-cell footprint grid.

    EN: Create a fine occupied-cell footprint from the selected input catalog;
    this can be used to correct aperture areas near survey edges or star-mask
    holes.
    ZH: 从筛选后的输入星表构建细网格有效 footprint，可用于修正 survey 边界或 star-mask 空洞附近的孔径面积。
    """

    if cell_deg is None or cell_deg <= 0:
        return None
    ra = catalog["ra"].to_numpy(float)
    dec = catalog["dec"].to_numpy(float)
    center_ra = float(np.median(ra))
    center_dec = float(np.median(dec))
    cos_dec = float(np.cos(np.radians(center_dec)))
    x = (ra - center_ra) * cos_dec
    y = dec - center_dec
    xmin = float(np.floor(np.min(x) / cell_deg) * cell_deg)
    ymin = float(np.floor(np.min(y) / cell_deg) * cell_deg)
    nx = int(np.ceil((np.max(x) - xmin) / cell_deg)) + 1
    ny = int(np.ceil((np.max(y) - ymin) / cell_deg)) + 1
    ix = np.floor((x - xmin) / cell_deg).astype(int)
    iy = np.floor((y - ymin) / cell_deg).astype(int)
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[iy, ix] = True
    cell_y, cell_x = np.where(occupied)
    return {
        "cell_deg": float(cell_deg),
        "center_ra": center_ra,
        "center_dec": center_dec,
        "cos_dec": cos_dec,
        "xmin": xmin,
        "ymin": ymin,
        "cell_x": xmin + (cell_x + 0.5) * cell_deg,
        "cell_y": ymin + (cell_y + 0.5) * cell_deg,
        "area_deg2": float(np.sum(occupied) * cell_deg * cell_deg),
    }


def aperture_footprint_area_deg2(
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    footprint_grid: dict[str, Any] | None,
) -> float | None:
    """Estimate aperture area intersected with the effective footprint.

    EN: Count fine footprint-grid cells whose centers fall inside the aperture
    and return the corresponding sky area in square degrees.
    ZH: 统计孔径内的有效 footprint 细网格中心数量，并返回对应的平方度面积。
    """

    if footprint_grid is None:
        return None
    cx = (center_ra - float(footprint_grid["center_ra"])) * float(footprint_grid["cos_dec"])
    cy = center_dec - float(footprint_grid["center_dec"])
    dx = np.asarray(footprint_grid["cell_x"], dtype=float) - cx
    dy = np.asarray(footprint_grid["cell_y"], dtype=float) - cy
    inside = dx * dx + dy * dy <= radius_deg * radius_deg
    return float(np.sum(inside) * float(footprint_grid["cell_deg"]) ** 2)


def build_redshift_centers(config: SliceConfig) -> np.ndarray:
    """Build redshift-slice centers.

    EN: Return the regularly spaced slice centers from z_min to z_max.
    ZH: 返回从 z_min 到 z_max 的规则红移切片中心。
    """

    return np.arange(config.z_min, config.z_max + 0.5 * config.z_step, config.z_step)


def slice_half_width(z_center: float, config: SliceConfig) -> float:
    """Compute the dynamic half-width of a redshift slice.

    EN: Use dz = factor * (1 + z_center), matching the manuscript blind-search
    definition.
    ZH: 使用 dz = factor * (1 + z_center)，与论文中的动态红移切片定义一致。
    """

    return float(config.half_width_factor * (1.0 + z_center))


def angular_scale_deg_per_mpc(cosmology: FlatLambdaCDM, z_center: float) -> float:
    """Compute angular size per physical Mpc.

    EN: Convert one physical Mpc at the slice redshift into degrees on the sky.
    ZH: 将切片红移处的 1 个物理 Mpc 转换为天空角尺度（degree）。
    """

    return float(cosmology.arcsec_per_kpc_proper(z_center).to(u.deg / u.Mpc).value)


def deg2_to_physical_mpc2(area_deg2: float, z_center: float, cosmology: FlatLambdaCDM) -> float:
    """Convert a sky area from square degrees to physical Mpc^2.

    EN: Use the angular-diameter distance at the slice redshift to convert a
    solid angle into a transverse physical area.
    ZH: 使用切片红移处的角径距离，将天空立体角面积转换为横向物理面积。
    """

    da_mpc = float(cosmology.angular_diameter_distance(z_center).value)
    return float(area_deg2 * (math.pi / 180.0) ** 2 * da_mpc**2)


def compute_candidate_overdensity(
    n_members: int,
    scale_mpc: float,
    z_center: float,
    n_slice_total: int,
    center_ra: float,
    center_dec: float,
    config: BlindSearchConfig,
    cosmology: FlatLambdaCDM,
    footprint_grid: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the aperture-based candidate overdensity proxy.

    EN: Evaluate delta_candidate = rho_aper / rho_bg - 1, where rho_aper is
    N_member divided by pi*(f*R_G)^2 and rho_bg is the total redshift-slice
    galaxy count divided by the configured tight-area footprint.
    ZH: 计算 delta_candidate = rho_aper / rho_bg - 1，其中 rho_aper 为 N_member 除以
    pi*(f*R_G)^2，rho_bg 为红移切片总星系数除以配置的 tight-area 面积。
    """

    area_deg2 = config.candidate_density.background_area_deg2
    aperture_factor = config.candidate_density.aperture_radius_factor
    aperture_radius_mpc = aperture_factor * scale_mpc
    full_aperture_area_mpc2 = math.pi * aperture_radius_mpc**2
    aperture_radius_deg = aperture_radius_mpc * angular_scale_deg_per_mpc(cosmology, z_center)
    effective_aperture_area_deg2 = aperture_footprint_area_deg2(
        center_ra,
        center_dec,
        aperture_radius_deg,
        footprint_grid,
    )
    if effective_aperture_area_deg2 is None or effective_aperture_area_deg2 <= 0:
        aperture_area_mpc2 = full_aperture_area_mpc2
        effective_aperture_area_deg2 = float("nan")
    else:
        aperture_area_mpc2 = deg2_to_physical_mpc2(effective_aperture_area_deg2, z_center, cosmology)
    if area_deg2 is None or area_deg2 <= 0 or n_slice_total <= 0 or aperture_area_mpc2 <= 0:
        return {
            "delta_candidate": float("nan"),
            "rho_aper_mpc2": float("nan"),
            "rho_bg_mpc2": float("nan"),
            "aperture_area_mpc2": float(aperture_area_mpc2),
            "aperture_area_effective_deg2": float(effective_aperture_area_deg2),
            "aperture_area_full_mpc2": float(full_aperture_area_mpc2),
            "background_area_mpc2": float("nan"),
        }

    background_area_mpc2 = deg2_to_physical_mpc2(float(area_deg2), z_center, cosmology)
    rho_aper = float(n_members) / aperture_area_mpc2
    rho_bg = float(n_slice_total) / background_area_mpc2 if background_area_mpc2 > 0 else float("nan")
    delta_candidate = rho_aper / rho_bg - 1.0 if np.isfinite(rho_bg) and rho_bg > 0 else float("nan")
    return {
        "delta_candidate": float(delta_candidate),
        "rho_aper_mpc2": float(rho_aper),
        "rho_bg_mpc2": float(rho_bg),
        "aperture_area_mpc2": float(aperture_area_mpc2),
        "aperture_area_effective_deg2": float(effective_aperture_area_deg2),
        "aperture_area_full_mpc2": float(full_aperture_area_mpc2),
        "background_area_mpc2": float(background_area_mpc2),
    }


def sigma_pixels_for_scale(
    scale_mpc: float,
    z_center: float,
    pixel_size_deg: float,
    cosmology: FlatLambdaCDM,
    config: MapConfig,
) -> float:
    """Convert a physical Gaussian scale to pixels.

    EN: Compute sigma_pix = R_G * s(z) / pixel_size and clip it to the
    configured numerical range.
    ZH: 计算 sigma_pix = R_G * s(z) / pixel_size，并将其截断到配置的数值范围内。
    """

    sigma_deg = scale_mpc * angular_scale_deg_per_mpc(cosmology, z_center)
    return float(np.clip(sigma_deg / pixel_size_deg, config.sigma_pixel_min, config.sigma_pixel_max))


def compute_density_maps(
    histogram: np.ndarray,
    sigma_pixels: float,
    map_config: MapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed counts, local background, overdensity, and significance.

    EN: Apply a cluster-scale Gaussian and a broader local-background Gaussian,
    then compute delta = (n_sm - n_bg) / n_bg and S = (n_sm - n_bg) / sqrt(n_bg).
    ZH: 应用团簇尺度高斯核和更宽的局部背景高斯核，并计算 delta 与 Poisson-like 显著性 S。
    """

    smoothed = gaussian_filter(histogram, sigma=sigma_pixels, mode="constant")
    background = gaussian_filter(
        histogram,
        sigma=sigma_pixels * map_config.background_sigma_factor,
        mode="constant",
    )
    background = np.maximum(background, map_config.background_floor)
    overdensity = (smoothed - background) / background
    significance = (smoothed - background) / np.sqrt(background)
    return smoothed, background, overdensity, significance


def refine_peak_position(
    ra_values: np.ndarray,
    dec_values: np.ndarray,
    grid_ra: float,
    grid_dec: float,
    radius_deg: float,
) -> tuple[float, float, int]:
    """Refine a grid peak using nearby galaxy positions.

    EN: Average galaxies within the configured aperture around the grid peak;
    if none are found, keep the grid center.
    ZH: 对网格峰附近孔径内的星系位置取平均；若没有星系，则保留网格中心。
    """

    cos_dec = math.cos(math.radians(grid_dec))
    dist_sq = ((ra_values - grid_ra) * cos_dec) ** 2 + (dec_values - grid_dec) ** 2
    near = dist_sq <= radius_deg**2
    if np.any(near):
        return float(np.mean(ra_values[near])), float(np.mean(dec_values[near])), int(np.sum(near))
    return float(grid_ra), float(grid_dec), 0


def extract_peaks_from_slice(
    ra_values: np.ndarray,
    dec_values: np.ndarray,
    z_center: float,
    n_slice_total: int,
    ra_bins: np.ndarray,
    dec_bins: np.ndarray,
    config: BlindSearchConfig,
    cosmology: FlatLambdaCDM,
    footprint_grid: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract raw overdensity peaks from one redshift slice.

    EN: Build the 2D count map for a slice, smooth it at each configured
    physical scale, select local maxima, and return raw peak records.
    ZH: 为一个红移切片构建二维计数图，在每个配置物理尺度上平滑，筛选局部极大值，并返回 raw peak 记录。
    """

    pixel_size_deg = config.density_map.pixel_size_arcmin / 60.0
    histogram, _, _ = np.histogram2d(ra_values, dec_values, bins=[ra_bins, dec_bins])
    peaks: list[dict[str, Any]] = []

    for scale_mpc in config.density_map.smoothing_scales_mpc:
        sigma_pix = sigma_pixels_for_scale(scale_mpc, z_center, pixel_size_deg, cosmology, config.density_map)
        _, _, delta_map, significance_map = compute_density_maps(histogram, sigma_pix, config.density_map)
        local_max = maximum_filter(significance_map, size=config.density_map.maximum_filter_size) == significance_map
        peak_mask = (
            local_max
            & (significance_map >= config.peaks.significance_min)
            & (delta_map >= config.peaks.overdensity_min)
        )
        scale_radius_deg = scale_mpc * angular_scale_deg_per_mpc(cosmology, z_center)
        refine_radius_deg = config.peaks.refinement_radius_factor * scale_radius_deg
        ix_values, iy_values = np.where(peak_mask)

        for ix, iy in zip(ix_values, iy_values):
            grid_ra = float(ra_bins[ix] + 0.5 * pixel_size_deg)
            grid_dec = float(dec_bins[iy] + 0.5 * pixel_size_deg)
            refined_ra, refined_dec, n_members = refine_peak_position(
                ra_values,
                dec_values,
                grid_ra,
                grid_dec,
                refine_radius_deg,
            )
            density_values = compute_candidate_overdensity(
                n_members,
                float(scale_mpc),
                float(z_center),
                int(n_slice_total),
                refined_ra,
                refined_dec,
                config,
                cosmology,
                footprint_grid,
            )
            peaks.append(
                {
                    "RA": refined_ra,
                    "Dec": refined_dec,
                    "z_cen": float(z_center),
                    "significance": float(significance_map[ix, iy]),
                    "delta": float(delta_map[ix, iy]),
                    "scale_mpc": float(scale_mpc),
                    "sigma_pixels": float(sigma_pix),
                    "n_members": int(n_members),
                    "n_slice_total": int(n_slice_total),
                    "background_area_deg2": (
                        np.nan
                        if config.candidate_density.background_area_deg2 is None
                        else float(config.candidate_density.background_area_deg2)
                    ),
                    **density_values,
                }
            )
    return peaks


def transverse_distance_mpc_h(
    ra1: float,
    dec1: float,
    z1: float,
    ra2: float,
    dec2: float,
    cosmology: FlatLambdaCDM,
) -> float:
    """Compute small-angle transverse separation in Mpc/h.

    EN: Use the angular-diameter distance at z1 and multiply by h to match the
    manuscript NMS threshold in Mpc/h.
    ZH: 使用 z1 处角径距离并乘以 h，从而匹配论文中以 Mpc/h 表示的 NMS 合并半径。
    """

    cos_dec = math.cos(math.radians(dec1))
    dra = math.radians(ra2 - ra1) * cos_dec
    ddec = math.radians(dec2 - dec1)
    sep_rad = math.hypot(dra, ddec)
    return float(sep_rad * cosmology.angular_diameter_distance(z1).value * (cosmology.H0.value / 100.0))


def candidate_index_key(ra: float, dec: float, z: float, config: MergeConfig) -> tuple[int, int, int]:
    """Compute the coarse index key used to accelerate NMS.

    EN: Assign a raw peak or kept candidate to a coarse redshift-sky cell.
    ZH: 将 raw peak 或已保留候选体分配到粗略的红移-天空网格。
    """

    return (
        int(math.floor(z / config.z_bin)),
        int(math.floor(ra / config.sky_cell_size_deg)),
        int(math.floor(dec / config.sky_cell_size_deg)),
    )


def nearby_index_keys(key: tuple[int, int, int], z_span: int) -> list[tuple[int, int, int]]:
    """List neighboring coarse cells to search during NMS.

    EN: Return nearby redshift-sky cells that may contain merge conflicts.
    ZH: 返回可能包含待合并冲突候选体的邻近粗网格。
    """

    z_bin, ra_bin, dec_bin = key
    keys = []
    for iz in range(z_bin - z_span, z_bin + z_span + 1):
        for ira in range(ra_bin - 1, ra_bin + 2):
            for idec in range(dec_bin - 1, dec_bin + 2):
                keys.append((iz, ira, idec))
    return keys


def initialize_kept_candidate(row: pd.Series) -> dict[str, Any]:
    """Create an internal kept-candidate record.

    EN: Convert one raw peak row into the mutable record used by greedy NMS.
    ZH: 将一条 raw peak 记录转换为 greedy NMS 使用的可变候选体记录。
    """

    return {
        "raw_id": int(row["raw_id"]),
        "RA": float(row["RA"]),
        "Dec": float(row["Dec"]),
        "z_cen": float(row["z_cen"]),
        "significance": float(row["significance"]),
        "delta": float(row["delta"]),
        "scale_mpc": float(row["scale_mpc"]),
        "n_members": int(row["n_members"]),
        "n_member_RA": float(row["RA"]),
        "n_member_Dec": float(row["Dec"]),
        "n_member_scale_mpc": float(row["scale_mpc"]),
        "n_member_z_cen": float(row["z_cen"]),
        "n_slice_total": int(row.get("n_slice_total", 0)),
        "background_area_deg2": float(row.get("background_area_deg2", np.nan)),
        "delta_candidate": float(row.get("delta_candidate", np.nan)),
        "rho_aper_mpc2": float(row.get("rho_aper_mpc2", np.nan)),
        "rho_bg_mpc2": float(row.get("rho_bg_mpc2", np.nan)),
        "aperture_area_mpc2": float(row.get("aperture_area_mpc2", np.nan)),
        "aperture_area_effective_deg2": float(row.get("aperture_area_effective_deg2", np.nan)),
        "aperture_area_full_mpc2": float(row.get("aperture_area_full_mpc2", np.nan)),
        "background_area_mpc2": float(row.get("background_area_mpc2", np.nan)),
        "detected_slices": {float(row["z_cen"])},
        "detected_scales": {float(row["scale_mpc"])},
        "z_min_det": float(row["z_cen"]),
        "z_max_det": float(row["z_cen"]),
    }


def merge_raw_peak_into_candidate(candidate: dict[str, Any], row: pd.Series) -> None:
    """Merge a lower-ranked raw peak into a kept candidate.

    EN: Preserve the highest-significance seed position while updating
    persistence metadata and maximum member count.
    ZH: 保留最高显著性 seed 的位置，同时更新持续性信息和最大成员数。
    """

    z_value = float(row["z_cen"])
    scale = float(row["scale_mpc"])
    candidate["detected_slices"].add(z_value)
    candidate["detected_scales"].add(scale)
    candidate["z_min_det"] = min(float(candidate["z_min_det"]), z_value)
    candidate["z_max_det"] = max(float(candidate["z_max_det"]), z_value)
    if int(row["n_members"]) > int(candidate["n_members"]):
        candidate["n_members"] = int(row["n_members"])
        candidate["n_member_RA"] = float(row["RA"])
        candidate["n_member_Dec"] = float(row["Dec"])
        candidate["n_member_scale_mpc"] = float(row["scale_mpc"])
        candidate["n_member_z_cen"] = z_value
        candidate["n_slice_total"] = int(row.get("n_slice_total", 0))
        candidate["background_area_deg2"] = float(row.get("background_area_deg2", np.nan))
        candidate["delta_candidate"] = float(row.get("delta_candidate", np.nan))
        candidate["rho_aper_mpc2"] = float(row.get("rho_aper_mpc2", np.nan))
        candidate["rho_bg_mpc2"] = float(row.get("rho_bg_mpc2", np.nan))
        candidate["aperture_area_mpc2"] = float(row.get("aperture_area_mpc2", np.nan))
        candidate["aperture_area_effective_deg2"] = float(row.get("aperture_area_effective_deg2", np.nan))
        candidate["aperture_area_full_mpc2"] = float(row.get("aperture_area_full_mpc2", np.nan))
        candidate["background_area_mpc2"] = float(row.get("background_area_mpc2", np.nan))


def merge_candidates_greedy_nms(
    raw_peaks: pd.DataFrame,
    config: MergeConfig,
    cosmology: FlatLambdaCDM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge duplicate raw peaks with greedy non-maximum suppression.

    EN: Sort raw peaks by decreasing significance and merge lower-ranked peaks
    into higher-ranked candidates when both the Mpc/h and redshift criteria are
    satisfied.
    ZH: 按显著性从高到低排序 raw peaks；若低排名峰同时满足 Mpc/h 距离和红移条件，则合并到高排名候选体。
    """

    if raw_peaks.empty:
        return pd.DataFrame(), pd.DataFrame()

    sorted_peaks = raw_peaks.sort_values("significance", ascending=False).reset_index(drop=True)
    max_z = float(sorted_peaks["z_cen"].max())
    z_span = int(math.ceil((config.redshift_factor * (1.0 + max_z)) / config.z_bin)) + 1
    kept: list[dict[str, Any]] = []
    spatial_index: dict[tuple[int, int, int], list[int]] = {}
    assignments: list[dict[str, Any]] = []

    for _, row in sorted_peaks.iterrows():
        row_key = candidate_index_key(float(row["RA"]), float(row["Dec"]), float(row["z_cen"]), config)
        candidate_indices: set[int] = set()
        for key in nearby_index_keys(row_key, z_span):
            candidate_indices.update(spatial_index.get(key, []))

        matched_index: int | None = None
        for idx in sorted(candidate_indices):
            existing = kept[idx]
            dz_limit = config.redshift_factor * (1.0 + float(existing["z_cen"]))
            if abs(float(row["z_cen"]) - float(existing["z_cen"])) > dz_limit:
                continue
            distance = transverse_distance_mpc_h(
                float(existing["RA"]),
                float(existing["Dec"]),
                float(existing["z_cen"]),
                float(row["RA"]),
                float(row["Dec"]),
                cosmology,
            )
            if distance <= config.spatial_radius_mpc_h:
                matched_index = idx
                break

        if matched_index is None:
            new_candidate = initialize_kept_candidate(row)
            kept.append(new_candidate)
            new_index = len(kept) - 1
            spatial_index.setdefault(row_key, []).append(new_index)
            assignments.append({"raw_id": int(row["raw_id"]), "cluster_index": new_index, "merge_type": "seed"})
        else:
            merge_raw_peak_into_candidate(kept[matched_index], row)
            assignments.append({"raw_id": int(row["raw_id"]), "cluster_index": matched_index, "merge_type": "merged"})

    final_rows: list[dict[str, Any]] = []
    cluster_to_id: dict[int, int] = {}
    next_id = 1
    for cluster_index, candidate in enumerate(kept):
        n_slices = len(candidate["detected_slices"])
        if n_slices < config.min_detected_slices:
            continue
        cluster_to_id[cluster_index] = next_id
        final_rows.append(
            {
                "ID": next_id,
                "RA": round(float(candidate["RA"]), 6),
                "Dec": round(float(candidate["Dec"]), 6),
                "z_peak": round(float(candidate["z_cen"]), 3),
                "significance": round(float(candidate["significance"]), 3),
                "delta": round(float(candidate["delta"]), 3),
                "delta_candidate": round(float(candidate["delta_candidate"]), 3),
                "best_scale_mpc": round(float(max(candidate["detected_scales"])), 3),
                "n_member_scale_mpc": round(float(candidate["n_member_scale_mpc"]), 3),
                "n_member_z_cen": round(float(candidate["n_member_z_cen"]), 3),
                "det_unique_slices": int(n_slices),
                "z_min_det": round(float(candidate["z_min_det"]), 3),
                "z_max_det": round(float(candidate["z_max_det"]), 3),
                "z_range": f"{candidate['z_min_det']:.2f}-{candidate['z_max_det']:.2f}",
                "scales_hit": "|".join(f"{x:g}" for x in sorted(candidate["detected_scales"])),
                "n_members": int(candidate["n_members"]),
                "n_member_RA": round(float(candidate["n_member_RA"]), 6),
                "n_member_Dec": round(float(candidate["n_member_Dec"]), 6),
                "n_slice_total": int(candidate["n_slice_total"]),
                "background_area_deg2": round(float(candidate["background_area_deg2"]), 6),
                "rho_aper_mpc2": float(candidate["rho_aper_mpc2"]),
                "rho_bg_mpc2": float(candidate["rho_bg_mpc2"]),
                "aperture_area_mpc2": float(candidate["aperture_area_mpc2"]),
                "aperture_area_effective_deg2": float(candidate["aperture_area_effective_deg2"]),
                "aperture_area_full_mpc2": float(candidate["aperture_area_full_mpc2"]),
                "background_area_mpc2": float(candidate["background_area_mpc2"]),
            }
        )
        next_id += 1

    assignment_df = pd.DataFrame(assignments)
    if not assignment_df.empty:
        assignment_df["ID"] = assignment_df["cluster_index"].map(cluster_to_id)
        assignment_df["survived"] = assignment_df["ID"].notna()
    final_df = pd.DataFrame(final_rows).sort_values("significance", ascending=False).reset_index(drop=True)
    return final_df, assignment_df


def scan_redshift_slices(
    catalog: pd.DataFrame,
    config: BlindSearchConfig,
    cosmology: FlatLambdaCDM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run raw peak extraction over all redshift slices.

    EN: Sort galaxies by photo-z, scan overlapping slices, and collect raw peak
    records plus per-slice statistics.
    ZH: 按 photo-z 排序星系，扫描重叠红移切片，并收集 raw peak 记录和逐切片统计。
    """

    pixel_size_deg = config.density_map.pixel_size_arcmin / 60.0
    ra_bins, dec_bins = build_sky_bins(catalog, pixel_size_deg)
    footprint_grid = build_footprint_grid(catalog, config.candidate_density.footprint_cell_deg)
    order = np.argsort(catalog["z"].to_numpy(float))
    z_values = catalog["z"].to_numpy(float)[order]
    ra_values = catalog["ra"].to_numpy(float)[order]
    dec_values = catalog["dec"].to_numpy(float)[order]
    centers = build_redshift_centers(config.slices)
    raw_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for idx, z_center in enumerate(centers, start=1):
        half_width = slice_half_width(float(z_center), config.slices)
        lo = np.searchsorted(z_values, z_center - half_width, side="left")
        hi = np.searchsorted(z_values, z_center + half_width, side="right")
        n_galaxies = int(hi - lo)
        peaks: list[dict[str, Any]] = []
        if n_galaxies >= config.slices.min_galaxies_per_slice:
            peaks = extract_peaks_from_slice(
                ra_values[lo:hi],
                dec_values[lo:hi],
                float(z_center),
                n_galaxies,
                ra_bins,
                dec_bins,
                config,
                cosmology,
                footprint_grid,
            )
            for peak in peaks:
                peak["slice_half_width"] = float(half_width)
                peak["slice_index"] = int(idx)
        raw_rows.extend(peaks)
        slice_rows.append(
            {
                "slice_index": int(idx),
                "z_cen": round(float(z_center), 4),
                "slice_half_width": round(float(half_width), 5),
                "n_galaxies": n_galaxies,
                "n_peaks": len(peaks),
            }
        )
        if idx == 1 or idx % 20 == 0 or idx == len(centers):
            print(
                f"slice {idx}/{len(centers)} z={z_center:.3f} "
                f"dz={half_width:.4f} galaxies={n_galaxies} peaks={len(peaks)}",
                flush=True,
            )

    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df.insert(0, "raw_id", np.arange(1, len(raw_df) + 1, dtype=int))
    return raw_df, pd.DataFrame(slice_rows)


def write_outputs(
    config: BlindSearchConfig,
    raw_peaks: pd.DataFrame,
    slice_stats: pd.DataFrame,
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Path]:
    """Write all blind-search output products.

    EN: Save raw peaks, slice statistics, merge assignments, final candidates,
    and a machine-readable run summary.
    ZH: 保存 raw peaks、切片统计、合并对应关系、最终候选体以及机器可读运行摘要。
    """

    out = config.output.directory
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_peaks": out / config.output.raw_peak_filename,
        "slice_stats": out / config.output.slice_filename,
        "merge_assignments": out / config.output.merge_filename,
        "candidates": out / config.output.candidate_filename,
        "summary": out / config.output.summary_filename,
    }
    raw_peaks.to_csv(paths["raw_peaks"], index=False)
    slice_stats.to_csv(paths["slice_stats"], index=False)
    assignments.to_csv(paths["merge_assignments"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def run_blind_search(config: BlindSearchConfig) -> dict[str, Path]:
    """Execute a full blind-search run.

    EN: Load the catalog, scan redshift slices, merge raw peaks, write outputs,
    and return the generated file paths.
    ZH: 加载星表、扫描红移切片、合并 raw peaks、写出结果，并返回生成文件路径。
    """

    cosmology = make_cosmology(config.cosmology)
    catalog = restrict_redshift_range(load_catalog(config.catalog), config.slices)
    if catalog.empty:
        raise RuntimeError("No galaxies remain after catalog cleaning and redshift cuts.")

    print(f"Loaded {len(catalog):,} galaxies for blind search.", flush=True)
    raw_peaks, slice_stats = scan_redshift_slices(catalog, config, cosmology)
    if raw_peaks.empty:
        raise RuntimeError("Blind search found no raw peaks; relax thresholds or inspect the input catalog.")

    candidates, assignments = merge_candidates_greedy_nms(raw_peaks, config.merge, cosmology)
    if candidates.empty:
        raise RuntimeError("No candidates survived NMS/persistence filtering; relax merge.min_detected_slices.")

    summary = {
        "input_catalog": str(config.catalog.path),
        "n_galaxies_used": int(len(catalog)),
        "n_slices": int(len(slice_stats)),
        "n_raw_peaks": int(len(raw_peaks)),
        "n_candidates": int(len(candidates)),
        "cosmology": {
            "H0": float(config.cosmology.h0),
            "h": float(config.cosmology.h0 / 100.0),
            "Omega_m": float(config.cosmology.omega_m),
            "Omega_Lambda": float(1.0 - config.cosmology.omega_m),
        },
        "candidate_density": {
            "definition": "delta_candidate = (N_member / (pi*(aperture_radius_factor*R_G)^2)) / (N_slice_total / S_tight) - 1",
            "background_area_deg2": config.candidate_density.background_area_deg2,
            "aperture_radius_factor": config.candidate_density.aperture_radius_factor,
            "footprint_cell_deg": config.candidate_density.footprint_cell_deg,
        },
        "top_candidate": candidates.iloc[0].to_dict(),
    }
    paths = write_outputs(config, raw_peaks, slice_stats, candidates, assignments, summary)
    print(f"Final candidates: {len(candidates):,}", flush=True)
    print(f"Candidate table: {paths['candidates']}", flush=True)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    EN: Define the command-line interface for running the configurable blind
    search.
    ZH: 定义运行可配置 blind search 的命令行接口。
    """

    parser = argparse.ArgumentParser(description="Run a configurable blind overdensity search.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML or JSON configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    EN: Load the configuration file and execute the blind-search pipeline.
    ZH: 加载配置文件并执行 blind-search pipeline。
    """

    args = parse_args(argv)
    config = build_config(load_mapping(args.config))
    run_blind_search(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
