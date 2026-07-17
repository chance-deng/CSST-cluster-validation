# Blindsearch Configuration Guide

这份配置文档对应 `blind_search.py` 和 `config.yaml`，目标是把原先 CSST blind cluster-finding pipeline 中的关键参数全部外置，使其可以用于不同输入星表、不同红移范围、不同平滑核和不同宇宙学参数。

## 运行方式

```bash
python3 /Users/dengcanze/Documents/CSST/blindsearch/blind_search.py \
  --config /Users/dengcanze/Documents/CSST/blindsearch/config.yaml
```

输出包括：

- `blindsearch_candidates.csv`：最终 NMS 合并后的候选体表。
- `blindsearch_raw_peaks.csv`：每个 redshift slice 和 smoothing scale 上的 raw peaks。
- `blindsearch_slice_stats.csv`：逐切片的星系数和 peak 数。
- `blindsearch_merge_assignments.csv`：raw peaks 到最终候选体的合并关系。
- `blindsearch_summary.json`：输入、宇宙学参数、候选体数量和 top candidate 摘要。

## 配置项说明

### `catalog`

输入星表与列名映射。

- `path`：输入星表路径，支持 FITS、CSV、TXT、ECSV。
- `ra_column`：赤经列名，单位 degree。
- `dec_column`：赤纬列名，单位 degree。
- `redshift_column`：photo-z 列名。
- `magnitude_column`：可选星等列名；若不做星等截断，设为 `null`。
- `magnitude_limit`：可选星等上限，例如 `22.0` 表示 `mag < 22.0`。

### `cosmology`

宇宙学参数用于把物理尺度转换为角尺度。

- `h0`：Hubble constant，单位 km/s/Mpc。
- `omega_m`：物质密度参数。

Jiutian/Planck 2018 默认值：

```yaml
cosmology:
  h0: 67.66
  omega_m: 0.3111
```

旧版早期测试若需复现 `FlatLambdaCDM(H0=70, Om0=0.3)`，可改为：

```yaml
cosmology:
  h0: 70.0
  omega_m: 0.3
```

### `slices`

红移切片设置。论文中的 fiducial 规则为：

\[
|z_{\rm phot}-z_{\rm s}| \le 0.03(1+z_{\rm s})
\]

- `z_min`、`z_max`：全局搜索红移范围。
- `z_step`：红移切片中心步长，fiducial 为 `0.01`。
- `half_width_factor`：动态切片半宽系数，fiducial 为 `0.03`。
- `min_galaxies_per_slice`：少于该星系数的切片不做 peak extraction。

### `density_map`

二维密度图与平滑核设置。

- `pixel_size_arcmin`：天空像素尺寸，论文 fiducial 为 `0.3` arcmin。
- `smoothing_scales_mpc`：团簇尺度 Gaussian kernel，单位 physical Mpc；论文 fiducial 为 `[0.4, 0.8, 1.2]`。
- `background_sigma_factor`：局部背景核相对于团簇核的放大倍数；论文定义为 `15.0`。
- `sigma_pixel_min`、`sigma_pixel_max`：`sigma_pix` 数值截断范围，避免极端红移处核过小或过大。
- `background_floor`：背景图下限，避免除零。
- `maximum_filter_size`：局部极大值筛选窗口，论文实现为 `5`，即 `5x5` pixels。

过密度定义为：

\[
\delta_{ij} =
\frac{n^{\rm sm}_{ij} - n^{\rm bg}_{ij}}
{n^{\rm bg}_{ij}} .
\]

显著性定义为：

\[
S_{ij} =
\frac{n^{\rm sm}_{ij} - n^{\rm bg}_{ij}}
{\sqrt{n^{\rm bg}_{ij}}}.
\]

这里的 `S` 只是经验排序统计量，不是完整统计似然。

### `peaks`

raw peak 筛选规则。

- `significance_min`：保留 raw peak 的最小 `S`，论文 fiducial 为 `0.2`。
- `overdensity_min`：保留 raw peak 的最小 `delta`，论文 fiducial 为 `0.5`。
- `refinement_radius_factor`：用 `factor * R_G` 内的星系平均位置修正峰中心，论文实现为 `1.5`。

### `candidate_density`

额外输出的孔径过密度参考量：

\[
\delta_{\rm candidate}
=
\frac{\rho_{\rm aper}-\rho_{\rm bg}}{\rho_{\rm bg}}
=
\frac{N_{\rm member}}
{\pi(fR_G)^2\rho_{\rm bg}}-1 .
\]

其中：

- \(N_{\rm member}\)：raw peak 周围 \(fR_G\) 孔径内的星系数；最终候选体取 NMS 合并 raw peaks 中的最大值对应记录。
- \(f\)：`aperture_radius_factor`，默认 `1.5`。
- \(R_G\)：该 peak 的 Gaussian smoothing scale，单位 physical Mpc。
- \(\rho_{\rm bg}=N_{\rm slice,total}/S_{\rm tight}\)：同一 redshift slice 内的总星系数除以 tight footprint 面积，转换到 physical Mpc\(^{-2}\)。

配置项：

- `background_area_deg2`：tight footprint 面积，单位 square degree。若设为 `null`，输出的 `delta_candidate` 为 `NaN`。
- `aperture_radius_factor`：孔径半径相对于 \(R_G\) 的倍数，默认 `1.5`。

这个量与 `delta` 不同：`delta` 是 Gaussian-smoothed peak pixel 相对于 broad local-background map 的过密度；`delta_candidate` 是 hard aperture richness 相对于全 slice tight-area 背景的过密度。

### `merge`

greedy NMS 合并规则。

- `spatial_radius_mpc_h`：空间合并半径，单位 physical Mpc/h；论文 fiducial 为 `1.0`。
- `redshift_factor`：红移合并条件为 `|z_1-z_2| <= redshift_factor * (1+z_1)`；论文 fiducial 为 `0.04`。
- `min_detected_slices`：最终候选体至少需要被多少个不同红移切片探测到；论文 fiducial 为 `2`。
- `z_bin`、`sky_cell_size_deg`：NMS 加速用粗索引参数，不改变科学定义。

### `output`

输出路径和文件名。

CSST 项目内建议把运行产物放到：

```yaml
output:
  directory: /Users/dengcanze/Documents/CSST/Codex/result/<your_run_name>
```

若后续上传 GitHub，可将默认输出路径改成相对路径，例如：

```yaml
output:
  directory: results/example_run
```

## 与历史版本的关系

这份干净版主要固化论文中正式描述的 blind-search 逻辑：

- 只使用 `RA, Dec, z_phot` 和可选星等截断；
- 不使用 truth cluster、halo label、red-sequence prior；
- 使用动态红移切片；
- 使用 Gaussian-smoothed count map 和 broad local-background map；
- 使用 `delta=(n_sm-n_bg)/n_bg`；
- 使用 `S=(n_sm-n_bg)/sqrt(n_bg)` 作为 peak ranking statistic；
- 使用 greedy NMS 合并相邻红移切片和多平滑尺度上的重复探测；
- 使用 Jiutian/Planck 2018 宇宙学参数作为默认配置。

旧的 PPM validation、cross-match、purity/completeness scan 属于后处理验证步骤，没有放进这个通用 blind-search 脚本中。
