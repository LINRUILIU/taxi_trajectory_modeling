# Unified Analysis Summary: selector_full_val

## Global Metrics
- 1/8: MAE=78.0982 m, RMSE=105.3778 m, P95=213.8129 m, Topology=1.69%
- 1/16: MAE=137.8786 m, RMSE=189.6296 m, P95=388.8840 m, Topology=3.93%

## Interpretation
- `shape_symmetric_m` is used as the reference path-similarity metric; lower is better.
- `metric_cheat` cases indicate low official MAE but still visibly mismatched geometry.
- `path_wrong` cases indicate both official error and route geometry are poor and deserve manual review first.

## Case Gallery
- Selected cases: 8 total, with 1/8=4 and 1/16=4.
- Each case figure overlays road network, known points, actual missing points, predicted missing points, actual path, and predicted path.
