# Field05 i<22.0+n_members>=7：PPM 输出参数阈值扫描

## 目的

测试 PPM 输出参数是否能在当前候选体集合上进一步提高 purity proxy，并记录 recovery 的损失。

## 指标定义

- covered true clusters 固定为 `1188`。
- recovery = matched true clusters / covered true clusters。
- purity proxy = 至少匹配到一个 true cluster 的 candidate 数 / 保留 candidate 数。
- candidate density = 保留 candidate 数 / covered true clusters。

## Baseline

- baseline：`i<22.0 + n_members>=7`，candidate `4590`，matched true clusters `956`，recovery `80.47%`，purity proxy `40.78%`，F1 `0.541`。
- 仅要求有有效 PPM 输出：candidate `4041`，matched true clusters `884`，recovery `74.41%`，purity proxy `41.18%`，F1 `0.530`。

## 最优 F1

| cut | n_candidates | matched_true_clusters | recovery | purity_proxy | candidate_density | f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline: i<22.0+n_members>=7 | 4590 | 956 | 0.8047 | 0.4078 | 3.8636 | 0.5413 |
| PPM_richness/(rmax^2*z_rms) >= 75.088 | 3394 | 826 | 0.6953 | 0.4393 | 2.8569 | 0.5384 |
| PPM_significance >= 2.4423 & PPM_richness >= 9 | 3209 | 794 | 0.6684 | 0.4506 | 2.7012 | 0.5383 |
| PPM_richness >= 7 | 3727 | 849 | 0.7146 | 0.4317 | 3.1372 | 0.5383 |
| PPM_richness >= 9 | 3476 | 814 | 0.6852 | 0.4430 | 2.9259 | 0.5381 |
| PPM_significance >= 2.4423 & PPM_richness >= 7 | 3413 | 824 | 0.6936 | 0.4389 | 2.8729 | 0.5376 |
| PPM_significance >= 2.269 & PPM_richness >= 9 | 3343 | 803 | 0.6759 | 0.4457 | 2.8140 | 0.5372 |
| PPM_richness >= 6 | 3841 | 863 | 0.7264 | 0.4259 | 3.2332 | 0.5370 |

## 在 recovery >= 80% 时 purity 最高

| cut | n_candidates | matched_true_clusters | recovery | purity_proxy | candidate_density | f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline: i<22.0+n_members>=7 | 4590 | 956 | 0.8047 | 0.4078 | 3.8636 | 0.5413 |

## 在 recovery >= 70% 时 purity 最高

| cut | n_candidates | matched_true_clusters | recovery | purity_proxy | candidate_density | f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPM_significance >= 2.269 & PPM_richness >= 7 | 3574 | 834 | 0.7020 | 0.4337 | 3.0084 | 0.5362 |
| PPM_richness >= 7 | 3727 | 849 | 0.7146 | 0.4317 | 3.1372 | 0.5383 |
| PPM_richness/rmax^2 >= 3.136 & PPM_z_rms <= 0.09405 | 3445 | 838 | 0.7054 | 0.4276 | 2.8998 | 0.5324 |
| PPM_richness/rmax^2 >= 4.002 & PPM_z_rms <= 0.09906 | 3513 | 838 | 0.7054 | 0.4273 | 2.9571 | 0.5322 |
| PPM_significance >= 2.269 & PPM_richness >= 6 | 3678 | 845 | 0.7113 | 0.4269 | 3.0960 | 0.5335 |
| PPM_richness/(rmax^2*z_rms) >= 61.174 | 3636 | 847 | 0.7130 | 0.4266 | 3.0606 | 0.5338 |
| PPM_richness >= 6 | 3841 | 863 | 0.7264 | 0.4259 | 3.2332 | 0.5370 |
| PPM_richness/rmax^2 >= 4.002 | 3698 | 849 | 0.7146 | 0.4254 | 3.1128 | 0.5333 |

## 在 recovery >= 50% 时 purity 最高

| cut | n_candidates | matched_true_clusters | recovery | purity_proxy | candidate_density | f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPM_significance >= 2.6151 & PPM_richness >= 18 | 1983 | 604 | 0.5084 | 0.4786 | 1.6692 | 0.4930 |
| PPM_significance >= 2.4423 & PPM_richness >= 18 | 2025 | 608 | 0.5118 | 0.4780 | 1.7045 | 0.4943 |
| PPM_significance >= 3.6009 & PPM_richness >= 11 | 1880 | 605 | 0.5093 | 0.4755 | 1.5825 | 0.4918 |
| PPM_significance >= 2.269 & PPM_richness >= 18 | 2053 | 611 | 0.5143 | 0.4754 | 1.7281 | 0.4941 |
| PPM_significance >= 3.6009 & PPM_richness >= 9 | 1939 | 623 | 0.5244 | 0.4750 | 1.6322 | 0.4985 |
| PPM_richness >= 18 | 2082 | 613 | 0.5160 | 0.4745 | 1.7525 | 0.4944 |
| PPM_significance >= 3.0933 & PPM_richness >= 13 | 2261 | 646 | 0.5438 | 0.4732 | 1.9032 | 0.5061 |
| PPM_significance >= 2.8866 & PPM_richness >= 13 | 2445 | 679 | 0.5715 | 0.4720 | 2.0581 | 0.5170 |

## 纯度最高的极端阈值

| cut | n_candidates | matched_true_clusters | recovery | purity_proxy | candidate_density | f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPM_richness/rmax^2 >= 13.007 & PPM_z_rms <= 0.03417 | 4 | 5 | 0.0042 | 1.0000 | 0.0034 | 0.0084 |
| PPM_richness/rmax^2 >= 14.007 & PPM_z_rms <= 0.03417 | 4 | 5 | 0.0042 | 1.0000 | 0.0034 | 0.0084 |
| PPM_richness/rmax^2 >= 16.008 & PPM_z_rms <= 0.03417 | 1 | 1 | 0.0008 | 1.0000 | 0.0008 | 0.0017 |
| PPM_significance >= 5.2171 & PPM_z_rms <= 0.03417 | 22 | 24 | 0.0202 | 0.8182 | 0.0185 | 0.0394 |
| PPM_richness/rmax^2 >= 11.006 & PPM_z_rms <= 0.03417 | 11 | 11 | 0.0093 | 0.8182 | 0.0093 | 0.0183 |
| PPM_significance >= 5.8001 & PPM_z_rms <= 0.03417 | 19 | 23 | 0.0194 | 0.7895 | 0.0160 | 0.0378 |
| PPM_significance >= 5.2171 & PPM_z_rms <= 0.06272 | 134 | 121 | 0.1019 | 0.7537 | 0.1128 | 0.1795 |
| PPM_significance >= 7.1566 & PPM_z_rms <= 0.03417 | 12 | 17 | 0.0143 | 0.7500 | 0.0101 | 0.0281 |

## 图

![PPM threshold scan](../figures/ppm_threshold_scan_recovery_purity.png)

## 初步结论

- PPM 参数能略微提高 purity，但不能在保持高 recovery 的同时把 purity 推到很高。
- 在 recovery >= 80% 的要求下，当前扫描没有找到优于 baseline 的 PPM 阈值；baseline purity proxy 仍为约 `40.8%`。
- 若允许 recovery 降到约 70%，最佳 PPM cut 的 purity proxy 约 `43.4%`。
- 若允许 recovery 降到约 50%，purity proxy 可以提升到约 `47.9%`，但相比之前单独调 `n_members` 的收益并不明显。
- 这说明当前 PPM 输出更适合做候选体的后验诊断或排序特征；如果目标是显著提高 purity，可能需要把 PPM 特征、颜色/星等/stellar mass 信息以及局部密度形态一起训练一个监督式 reranker。
