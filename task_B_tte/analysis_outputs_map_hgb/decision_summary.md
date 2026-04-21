# Task B Analysis Summary

## Inputs
- Prediction: task_B_tte\pred_val_map_hgb.pkl
- Ground Truth: task_B_tte\val_gt.pkl
- Input (for bucket/context): task_B_tte\val_input.pkl

## Official Metrics
| count | MAE(s) | RMSE(s) | MAPE(%) | bias(s) | P90_abs_err(s) | P95_abs_err(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 16582 | 16.3995 | 25.5203 | 1.4117 | -0.4112 | 33.1762 | 45.4838 |

## Coverage
- matched=16582 / gt=16582 / pred=16582 / missing_pred=0 / extra_pred=0

## Delta vs Reference
- Delta MAE: +0.0119 s
- Delta RMSE: +0.0360 s
- Delta MAPE: +0.0012 %

## Judgment
- Heuristic grade: strong
- Worst travel-time bucket by MAPE: 10-20m (MAPE=1.4851%, count=10391)
- Best travel-time bucket by MAPE: 30-45m (MAPE=1.2076%, count=1210)

## Next Actions
- Keep this file as the milestone log anchor for future runs.
- Compare new runs via --reference-metrics and milestone trend plots.
