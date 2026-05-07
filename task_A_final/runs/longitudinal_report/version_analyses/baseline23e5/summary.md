# Unified Analysis Summary: baseline23e5

## Global Metrics
- 1/8: MAE=84.1953 m, RMSE=112.8769 m, P95=229.3611 m, Topology=3.30%
- 1/16: MAE=147.4253 m, RMSE=200.7644 m, P95=412.9867 m, Topology=7.36%

## Interpretation
- `shape_symmetric_m` is used as the reference path-similarity metric; lower is better.
- `metric_cheat` cases indicate low official MAE but still visibly mismatched geometry.
- `path_wrong` cases indicate both official error and route geometry are poor and deserve manual review first.

## Case Gallery
- Selected cases: 8 total, with 1/8=4 and 1/16=4.
- Each case figure overlays road network, known points, actual missing points, predicted missing points, actual path, and predicted path.
