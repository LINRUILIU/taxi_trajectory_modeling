# Task B Analysis Summary

## Inputs
- Prediction: task_B_tte\pred_val_phase4_residual_ensemble.pkl
- Ground Truth: task_B_tte\val_gt.pkl
- Input (for bucket/context): task_B_tte\val_input.pkl

## Official Metrics
| count | MAE(s) | RMSE(s) | MAPE(%) | bias(s) | P90_abs_err(s) | P95_abs_err(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 16582 | 16.3747 | 25.4080 | 1.4102 | -0.1566 | 33.2211 | 44.8328 |

## Coverage
- matched=16582 / gt=16582 / pred=16582 / missing_pred=0 / extra_pred=0

## Delta vs Reference
- Delta MAE: -0.0130 s
- Delta RMSE: -0.0763 s
- Delta MAPE: -0.0003 %

## Judgment
- Heuristic grade: strong
- Worst travel-time bucket by MAPE: 10-20m (MAPE=1.4843%, count=10391)
- Best travel-time bucket by MAPE: 30-45m (MAPE=1.1964%, count=1210)

## Next Actions
- Keep this file as the milestone log anchor for future runs.
- Compare new runs via --reference-metrics and milestone trend plots.
