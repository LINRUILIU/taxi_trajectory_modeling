# Unified Analysis Summary: baseline1

## Global Metrics
- 1/8: MAE=92.0430 m, RMSE=121.8241 m, P95=247.7450 m, Topology=11.73%
- 1/16: MAE=170.3906 m, RMSE=224.5509 m, P95=460.0388 m, Topology=21.33%

## Interpretation
- `shape_symmetric_m` is used as the reference path-similarity metric; lower is better.
- `metric_cheat` cases indicate low official MAE but still visibly mismatched geometry.
- `path_wrong` cases indicate both official error and route geometry are poor and deserve manual review first.

## Case Gallery
- Selected cases: 8 total, with 1/8=4 and 1/16=4.
- Each case figure overlays road network, known points, actual missing points, predicted missing points, actual path, and predicted path.
