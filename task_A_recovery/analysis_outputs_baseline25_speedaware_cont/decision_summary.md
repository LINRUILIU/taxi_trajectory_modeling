# Task A Visualization and Error Analysis Summary

## Global Metrics
- 1/8: MAE=83.3596 m, RMSE=111.5044 m, P95=224.9050 m
- 1/16: MAE=145.3025 m, RMSE=195.6828 m, P95=399.2673 m
- 1/8 topology_violation_rate=0.0507 (thr=35.0m, n=30000)
- 1/16 topology_violation_rate=0.1219 (thr=35.0m, n=30000)

## Speed Bucket Thresholds
- 1/8 speed_var quantiles: q33=0.9245, q66=2.8429 (m/s)
- 1/16 speed_var quantiles: q33=0.9250, q66=2.8430 (m/s)

## Trigger Checks
- Curvature trigger 1/8 (sharp >= 1.5x straight): True
- Curvature trigger 1/16 (sharp >= 1.5x straight): False
- Long-gap trigger 1/8 (5-7 >= 1.3x 1-2): True
- Long-gap trigger 1/16 (9-15 >= 1.3x 1-4): True

## Suggested Next Step
- Priority 1: add spline or smoothing baseline, then evaluate map-aware correction for turning segments.
- Priority 2: focus on long-gap reconstruction; if spline is still weak, evaluate route-constrained methods.
- Keep linear baseline as diagnostic lower bound for all future comparisons.
