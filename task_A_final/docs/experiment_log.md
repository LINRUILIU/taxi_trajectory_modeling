# Experiment Log

## Baseline Anchor: `b28_compat_full`

- Date: 2026-05-01
- Source run: [runs/b28_compat_full](/D:/student_release/student_release/task_A_final/runs/b28_compat_full)
- Role: current new-pipeline baseline anchor and temporary final config target

### Metrics

- `1/8`: MAE `81.7358`, RMSE `110.6424`, P95 `224.0972`, topology violation `2.61%`
- `1/16`: MAE `142.1400`, RMSE `195.7595`, P95 `401.1618`, topology violation `4.84%`

### Notes

- MAE compatibility passed against legacy `baseline28_turncurve`.
- Prediction files are not byte-identical to legacy outputs, so keep this as metric-level anchor rather than file-level anchor.
- Further optimization should compare against this run first, not against `pchip_only`.

## Promoted: `selector_oof` -> `selector_full_val`

- Date: 2026-05-01
- Source run: [runs/selector_oof](/D:/student_release/student_release/task_A_final/runs/selector_oof)
- Full-val run: [runs/selector_full_val](/D:/student_release/student_release/task_A_final/runs/selector_full_val)
- Thresholds: `1/8 -> 0.50`, `1/16 -> 0.50`
- Status: **promoted** - full-val training completed, standalone sanity passed

### OOF Delta vs `b28_compat_full`

- `1/8`: MAE `-3.3076m`, P95 `-9.4642m`
- `1/16`: MAE `-3.2137m`, P95 `-9.5261m`

### Full-Val Results (`selector_full_val`)

- `1/8`: MAE `78.10m`, RMSE `105.38m`, P95 `213.81m`, topology violation `1.69%`
- `1/16`: MAE `137.88m`, RMSE `189.63m`, P95 `388.88m`, topology violation `3.93%`

### Notes

- Full-val training with fixed threshold 0.50 completed successfully.
- Selection rate: 1/8 ~14.9%, 1/16 ~10.9% (within expected band).
- drift_note: "ok" for both datasets.
- Ready for final submission.
