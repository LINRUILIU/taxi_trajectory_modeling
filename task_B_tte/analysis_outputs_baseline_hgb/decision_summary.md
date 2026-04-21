# Task B Analysis Summary

## Inputs
- Prediction: task_B_tte\pred_val_baseline_hgb.pkl
- Ground Truth: task_B_tte\val_gt.pkl
- Input (for bucket/context): task_B_tte\val_input.pkl

## Official Metrics
| count | MAE(s) | RMSE(s) | MAPE(%) | bias(s) | P90_abs_err(s) | P95_abs_err(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 16582 | 16.3876 | 25.4844 | 1.4105 | -0.3975 | 33.2550 | 45.0101 |

## Coverage
- matched=16582 / gt=16582 / pred=16582 / missing_pred=0 / extra_pred=0

## Judgment
- Heuristic grade: strong
- Worst travel-time bucket by MAPE: 10-20m (MAPE=1.4836%, count=10391)
- Best travel-time bucket by MAPE: 30-45m (MAPE=1.2086%, count=1210)

## Next Actions
- Keep this file as the milestone log anchor for future runs.
- Compare new runs via --reference-metrics and milestone trend plots.
