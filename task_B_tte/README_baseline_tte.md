# Task B Baseline ETA

This folder now includes a first runnable baseline implementation:
- `baseline_tte.py`
- `analyze_tte.py`

## 1) Check val alignment (optional but recommended)

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py check-align --limit 5000
```

## 2) Train baseline on full ds15 train and evaluate on task_B val

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py train \
  --hgb-max-iter 450 \
  --model-out task_B_tte/model_baseline_hgb.pkl \
  --metrics-out task_B_tte/metrics_baseline_hgb.json \
  --val-pred-out task_B_tte/pred_val_baseline_hgb.pkl
```

## 3) Evaluate a prediction file

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py evaluate \
  --pred task_B_tte/pred_val_baseline_hgb.pkl \
  --gt task_B_tte/val_gt.pkl \
  --metrics-out task_B_tte/metrics_from_pred_baseline_hgb.json
```

## 4) Generate submission for test_input.pkl (classroom test)

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py predict \
  --model-in task_B_tte/model_baseline_hgb.pkl \
  --input task_B_tte/test_input.pkl \
  --output task_B_tte/submission_test_baseline_hgb.pkl
```

## 5) Run analysis + visualization + milestone recording

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/analyze_tte.py \
  --pred task_B_tte/pred_val_baseline_hgb.pkl \
  --gt task_B_tte/val_gt.pkl \
  --input task_B_tte/val_input.pkl \
  --output-dir task_B_tte/analysis_outputs_baseline_hgb \
  --milestone-name baseline_hgb_v1
```

Main outputs:
- `global_metrics.json`
- `bucket_summary.csv`
- `decision_summary.md`
- `top_error_cases.csv`
- plot files (`scatter_pred_vs_gt.png`, `abs_error_hist.png`, `residual_vs_gt.png`, `bucket_*.png`)

Milestone outputs:
- `task_B_tte/analysis_outputs_milestones/milestone_metrics.csv`
- `task_B_tte/analysis_outputs_milestones/milestone_summary.md`
- `task_B_tte/analysis_outputs_milestones/milestone_trend.png`

## 6) Compare against a previous metrics file

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/analyze_tte.py \
  --pred task_B_tte/pred_val_new.pkl \
  --gt task_B_tte/val_gt.pkl \
  --input task_B_tte/val_input.pkl \
  --output-dir task_B_tte/analysis_outputs_new \
  --reference-metrics task_B_tte/analysis_outputs_baseline_hgb/global_metrics.json \
  --milestone-name new_run_v2
```

## 7) Optional map-aware experiment (Phase 3)

Train map-aware variant:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py train \
  --osm map \
  --map-cache task_B_tte/map_segments_cache.pkl \
  --model-out task_B_tte/model_map_hgb.pkl \
  --metrics-out task_B_tte/metrics_map_hgb.json \
  --val-pred-out task_B_tte/pred_val_map_hgb.pkl
```

Analyze + milestone compare vs baseline:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/analyze_tte.py \
  --pred task_B_tte/pred_val_map_hgb.pkl \
  --gt task_B_tte/val_gt.pkl \
  --input task_B_tte/val_input.pkl \
  --output-dir task_B_tte/analysis_outputs_map_hgb \
  --reference-metrics task_B_tte/analysis_outputs_baseline_hgb/global_metrics.json \
  --milestone-name map_hgb_v1
```

Predict with a model that was trained with map features:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/baseline_tte.py predict \
  --model-in task_B_tte/model_map_hgb.pkl \
  --input task_B_tte/test_input.pkl \
  --output task_B_tte/submission_test_map_hgb.pkl \
  --osm map \
  --map-cache task_B_tte/map_segments_cache.pkl
```

## 8) Phase 3 round-2 map screening (recommended before full map run)

Run map parameter screening with a no-map reference:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/phase3_round2_tune_map.py \
  --train-limit 24000 \
  --val-limit 6000 \
  --hgb-max-iter 260 \
  --osm map \
  --map-cache task_B_tte/map_segments_cache.pkl \
  --output-dir task_B_tte/phase3_round2_screen
```

Round-2 artifacts:
- `task_B_tte/phase3_round2_screen/round2_results.csv`
- `task_B_tte/phase3_round2_screen/round2_results.json`
- `task_B_tte/phase3_round2_screen/round2_summary.md`

Current round-2 conclusion (2026-04-20):
- best map case still worse than no-map reference on MAE/RMSE/MAPE.
- default production path remains `baseline_hgb_v1` (no-map).

## 9) Phase 4 residual + ensemble (current best)

Train full phase4 model:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/phase4_residual_ensemble.py train \
  --model-out task_B_tte/model_phase4_residual_ensemble.pkl \
  --metrics-out task_B_tte/metrics_phase4_residual_ensemble.json \
  --val-pred-out task_B_tte/pred_val_phase4_residual_ensemble.pkl
```

Analyze and append milestone:

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/analyze_tte.py \
  --pred task_B_tte/pred_val_phase4_residual_ensemble.pkl \
  --gt task_B_tte/val_gt.pkl \
  --input task_B_tte/val_input.pkl \
  --output-dir task_B_tte/analysis_outputs_phase4_residual_ensemble \
  --reference-metrics task_B_tte/analysis_outputs_baseline_hgb/global_metrics.json \
  --milestone-name phase4_residual_ensemble_v1
```

Generate test submission (recommended default):

```powershell
d:/student_release/student_release/.venv/Scripts/python.exe task_B_tte/phase4_residual_ensemble.py predict \
  --model-in task_B_tte/model_phase4_residual_ensemble.pkl \
  --input task_B_tte/test_input.pkl \
  --output task_B_tte/submission_test_phase4_residual_ensemble.pkl
```

Current best milestone (2026-04-20):
- `phase4_residual_ensemble_v1`
- MAE=16.3747, RMSE=25.4080, MAPE=1.4102
- relative to `baseline_hgb_v1`: MAE -0.0130, RMSE -0.0763, MAPE -0.0003

Output format matches required schema:
- list of dict
- each dict has: `traj_id`, `travel_time`

## Notes

- The model only uses trajectory coordinates and departure timestamp features.
- It does not use middle timestamps during inference feature extraction.
- Default train cleaning thresholds:
  - travel_time in [60, 7200] seconds
  - avg speed in [3, 120] km/h
- To run quick smoke training, add:

```powershell
--train-limit 6000 --val-limit 3000 --hgb-max-iter 180
```
