# Report Audit

This audit records the main factual checks performed against the local project files before the release split.

## Accurate Or Directly Verified

- Dataset scale and metadata match `作业说明.txt`: `data_org/train.pkl` and `data_ds15/train.pkl` contain 132,657 trajectories; validation contains 16,582 trajectories; coordinates are WGS-84; `data_ds15` is approximately 15-second sampled.
- Task B Phase4 metrics match `task_B_tte/metrics_phase4_residual_ensemble.json` and `task_B_tte/analysis_outputs_phase4_residual_ensemble/global_metrics.json`: MAE `16.3747s`, RMSE `25.4080s`, MAPE `1.4102%`.
- Task B map-aware degradation matches `task_B_tte/metrics_map_hgb.json`: MAE `16.3995s`, RMSE `25.5203s`, MAPE `1.4117%`.
- Task A final metrics match `task_A_final/runs/selector_full_val/analysis_unified/global_metrics.json`: MAE `78.10m / 137.88m`, RMSE `105.38m / 189.63m`, topology violation `1.69% / 3.93%`.
- Task A shape metrics match `task_A_final/runs/longitudinal_report/longitudinal_metrics.csv`: `shape_symmetric_mean_m` improves from `32.18 / 46.53` for `b28_compat` to `30.84 / 44.67` for `final`.
- Historical player-study Framework A/B/C metrics match `task_A_recovery/game_outputs/comparative_eval_report_20260421/framework_comparison.json`.
- The Markdown report's local links and images were checked for existence after restructuring.

## Corrected In The Report

- Task A input description previously said known coordinates were set to NaN. Correct statement: known points keep coordinates; missing points are NaN and must be recovered.
- The Task A final narrative previously over-emphasized framework construction. It now states that `task_A_final` also served the final metric sprint through `route_projection` and `selector_mix`.
- Historical interactive game and Framework A/B/C discussion previously occupied full main chapters. They are now an appendix because they are BL28-stage historical evidence, not a formal final-version human evaluation.
- The selector evidence boundary is now explicit: OOF supports the direction and threshold sanity; full-val artifacts support deployment sanity, but full-val metrics after selector training are not claimed as fully unbiased estimates.

## Statements That Should Stay Calibrated

- Comparisons to external papers are methodology-level context only. They are not strict same-dataset leaderboard claims.
- `shape_symmetric_m` is a reference metric for path geometry, not a replacement for the official point-level MAE/RMSE.
- The player-study "human upper bound" only applies under Framework B/C on the 76 historical BL28 cases.
- The final Task A model does not solve every hard case; remaining path-wrong cases are explicitly shown in longitudinal and unified analysis outputs.

## Evidence Files

These evidence files are retained on `release/full-artifacts`. The rendered report figures used by all release branches are copied into `docs/report_figures/`.

- `task_A_final/runs/selector_full_val/analysis_unified/global_metrics.json`
- `task_A_final/runs/selector_full_val/analysis_unified/quadrant_summary_global.json`
- `task_A_final/runs/longitudinal_report/longitudinal_metrics.csv`
- `task_A_final/runs/selector_full/train_full_report.json`
- `task_B_tte/metrics_phase4_residual_ensemble.json`
- `task_B_tte/analysis_outputs_phase4_residual_ensemble/global_metrics.json`
- `task_A_recovery/game_outputs/comparative_eval_report_20260421/framework_comparison.json`
