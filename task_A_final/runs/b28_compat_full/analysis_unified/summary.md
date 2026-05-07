# Unified Analysis Summary: b28_compat_full

## Global Metrics
- 1/8: MAE=81.7358 m, RMSE=110.6424 m, P95=224.0972 m, Topology=2.61%
- 1/16: MAE=142.1400 m, RMSE=195.7595 m, P95=401.1618 m, Topology=4.84%

## Interpretation
- `shape_symmetric_m` is used as the reference path-similarity metric; lower is better.
- `metric_cheat` cases indicate low official MAE but still visibly mismatched geometry.
- `path_wrong` cases indicate both official error and route geometry are poor and deserve manual review first.

## Case Gallery
- Selected cases: 8 total, with 1/8=4 and 1/16=4.
- Each case figure overlays road network, known points, actual missing points, predicted missing points, actual path, and predicted path.
