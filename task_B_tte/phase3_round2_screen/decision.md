# Phase 3 Round-2 Decision

## Screening Setup
- train_limit: 24000
- val_limit: 6000
- hgb_max_iter: 260
- no-map reference + 7 map-aware variants

## Result
- no-map reference: MAE=16.7825, RMSE=24.4845, MAPE=1.5475
- best map case (`map_r90_p18_t1545`): MAE=16.8581, RMSE=24.5143, MAPE=1.5544
- delta (best map - no-map):
  - MAE: +0.0756
  - RMSE: +0.0297
  - MAPE: +0.0069

## Decision
- Phase 3 round-2 still shows no map gain.
- Following project rule, drop map as the mainline for Task B.
- Keep `baseline_hgb_v1` as default production model and continue with Phase 4 (residual/ensemble route).
