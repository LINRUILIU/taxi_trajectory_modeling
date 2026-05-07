# Longitudinal Analysis Summary

## Final Takeaway
- `final` (selector mix) improves over `b28_compat` on MAE from 81.74m to 78.10m on 1/8, and from 142.14m to 137.88m on 1/16.
- Topology violation also drops from 2.61% to 1.69% on 1/8, and from 4.84% to 3.93% on 1/16.
- Mean reference path-similarity (`shape_symmetric_m`) also improves from 32.18m to 30.84m on 1/8, and from 46.53m to 44.67m on 1/16.

## Reading Guide
- `baseline1` is the no-road-network geometric lower bound.
- `baseline23e5` captures the first strong jump from HMM plus topology awareness.
- `b28_compat` is the final-pipeline anchor for legacy baseline28 behavior.
- `final` is the promoted selector-based release candidate.

## Case Pool
- Selected longitudinal cases: 6
- Categories cover improvement showcase, topology rescue, and remaining hard cases for both 1/8 and 1/16.
