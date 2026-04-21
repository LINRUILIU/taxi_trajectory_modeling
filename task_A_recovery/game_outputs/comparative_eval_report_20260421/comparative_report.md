# Player vs Baseline 综合评估报告

## 1) 数据与问题定义

- 评估对象: player_study_20260420_220213_r8 + player_study_20260420_224456_r16
- 总case: 76 (dataset8=38, dataset16=38)
- 目标: 比较 player / baseline23 / baseline28 在三种评价框架下的 MAE/RMSE 与胜率

## 2) 三种评价框架

1. Framework A (原始): 缺失索引逐点对齐比较。
2. Framework B: GT缺失点到预测整条轨迹最小距离。
3. Framework C: 每个gap内先均匀重采样到missing_count，再逐点比较。

## 3) 完整度诊断（解释为何A与B/C差异巨大）

- case_count: 76
- complete_cases: 1 ; incomplete_cases: 75
- total_missing: 5481 ; filled_missing: 2484 ; unfilled_missing: 2997
- filled_ratio: 0.4532 ; unfilled_ratio: 0.5468

结论: 原始记录里有大量未填缺失点，Framework A 会显著放大该惩罚；B/C 更关注轨迹形状本身。

## 4) Overall 对比（Weighted）

| Framework | Player MAE | Player RMSE | B23 MAE | B28 MAE | Player vs B23 胜率 | Player vs B28 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| A 索引逐点 | 308.7306 | 429.4691 | 117.0660 | 112.9521 | 0.0132 | 0.0000 |
| B 点到轨迹最短距 | 20.3295 | 58.3367 | 30.8133 | 27.1909 | 0.7632 | 0.7105 |
| C 均匀重采样后逐点 | 111.6913 | 156.5270 | 119.6088 | 115.1641 | 0.6842 | 0.5921 |

## 5) 关键变化量（Overall）

- Player MAE: A -> B: 308.7306 -> 20.3295 (下降 93.42%)
- Player MAE: A -> C: 308.7306 -> 111.6913 (下降 63.82%)
- B23 MAE: A -> B: 117.0660 -> 30.8133 (下降 73.68%)
- B23 MAE: A -> C: 117.0660 -> 119.6088 (下降 -2.17%)
- B28 MAE: A -> B: 112.9521 -> 27.1909 (下降 75.93%)
- B28 MAE: A -> C: 112.9521 -> 115.1641 (下降 -1.96%)

解释: baseline23/28 在 B/C 框架下也显著改善，说明它们同样存在时间相位/索引错位，但轨迹几何形状本身较接近GT。

## 6) 分数据集结论

### Framework A
- dataset=8: player MAE=206.4826, b23=81.8514, b28=79.1216, win_vs_b23=0.0263, win_vs_b28=0.0000
- dataset=16: player MAE=390.1670, b23=145.1131, b28=139.8967, win_vs_b23=0.0000, win_vs_b28=0.0000

### Framework B
- dataset=8: player MAE=13.8589, b23=19.1217, b28=18.3830, win_vs_b23=0.8158, win_vs_b28=0.7895
- dataset=16: player MAE=25.4831, b23=40.1252, b28=34.2060, win_vs_b23=0.7105, win_vs_b28=0.6316

### Framework C
- dataset=8: player MAE=82.9524, b23=83.8020, b28=82.5118, win_vs_b23=0.6053, win_vs_b28=0.5789
- dataset=16: player MAE=134.5806, b23=148.1274, b28=141.1703, win_vs_b23=0.7632, win_vs_b28=0.6053

## 7) 评价框架建议

1. 若要评估“严格时序对齐重建能力”，用 Framework A。
2. 若要评估“玩家描绘出的路线几何质量”，用 Framework B。
3. 若要兼顾“形状+gap内均匀时序”并保持与missing_count一致，推荐 Framework C 作为主指标。
4. 报告展示建议同时给 A 与 C：A 反映完整重建能力，C 反映交互描绘质量。

