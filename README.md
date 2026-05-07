# Taxi Trajectory Modeling

This repository contains a course project on Xi'an taxi GPS trajectories. It has two independent tasks:

- Task A: trajectory recovery from sparse GPS observations.
- Task B: travel time estimation from a complete path and departure time.

The current release uses `task_A_final` as the reproducible Task A pipeline. `task_A_recovery` keeps the historical exploration and the player-study analysis used in the report.

## Environment

Use Python 3.10+; the project was last run with Python 3.13 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`pygame` is only needed for the historical interactive labeling tool:

```powershell
.\.venv\Scripts\python.exe -m pip install pygame
```

## Data Layout

Expected root-level data and map files:

```text
data_org/train.pkl
data_org/val.pkl
data_ds15/train.pkl
data_ds15/val.pkl
map
task_A_final/val_input_8.pkl
task_A_final/val_input_16.pkl
task_A_final/val_gt.pkl
task_B_tte/val_input.pkl
task_B_tte/val_gt.pkl
```

The `release/code-lite` branch intentionally does not include these files. Place the required data and `map` file at the paths above before running the reproduction commands.

Classroom test files should be placed as:

```text
task_A_final/test_input_8.pkl
task_A_final/test_input_16.pkl
task_B_tte/test_input.pkl
```

## Task A Reproduction

Run from the repository root.

### 1. Build the final-pipeline BL28 anchor

```powershell
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_b28_compat_8.yaml --input task_A_final/val_input_8.pkl --out task_A_final/runs/b28_compat_full/pred_8.pkl
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_b28_compat_16.yaml --input task_A_final/val_input_16.pkl --out task_A_final/runs/b28_compat_full/pred_16.pkl
python task_A_final/scripts/run_analyze.py --input-8 task_A_final/val_input_8.pkl --input-16 task_A_final/val_input_16.pkl --pred-8 task_A_final/runs/b28_compat_full/pred_8.pkl --pred-16 task_A_final/runs/b28_compat_full/pred_16.pkl --gt task_A_final/val_gt.pkl --out-dir task_A_final/runs/b28_compat_full/analysis
```

### 2. Build route-projection candidates and debug features

```powershell
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_route_projection_8.yaml --input task_A_final/val_input_8.pkl --out task_A_final/runs/route_projection_full/pred_8.pkl
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_route_projection_16.yaml --input task_A_final/val_input_16.pkl --out task_A_final/runs/route_projection_full/pred_16.pkl
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_route_projection_8.yaml --input task_A_final/val_input_8.pkl --out task_A_final/runs/route_projection_full_debug/pred_8.pkl --debug-gap-csv task_A_final/runs/route_projection_full_debug/debug_8.csv
python task_A_final/scripts/run_predict.py --config task_A_final/configs/exp_route_projection_16.yaml --input task_A_final/val_input_16.pkl --out task_A_final/runs/route_projection_full_debug/pred_16.pkl --debug-gap-csv task_A_final/runs/route_projection_full_debug/debug_16.csv
python task_A_final/scripts/run_analyze.py --input-8 task_A_final/val_input_8.pkl --input-16 task_A_final/val_input_16.pkl --pred-8 task_A_final/runs/route_projection_full/pred_8.pkl --pred-16 task_A_final/runs/route_projection_full/pred_16.pkl --gt task_A_final/val_gt.pkl --out-dir task_A_final/runs/route_projection_full/analysis
```

### 3. Train and promote the selector

```powershell
python task_A_final/scripts/train_gap_selector.py --mode all
python task_A_final/scripts/train_gap_selector.py --mode train-full --out-dir task_A_final/runs/selector_full
```

`selector_oof` is used to choose and sanity-check the fixed threshold. `selector_full` is the deployment model trained on the available validation split.

### 4. Generate final validation predictions

```powershell
python task_A_final/scripts/run_predict.py --config task_A_final/configs/final_8.yaml --input task_A_final/val_input_8.pkl --out task_A_final/runs/selector_full_val/pred_8.pkl --debug-gap-csv task_A_final/runs/selector_full_val/selector_decisions_8.csv
python task_A_final/scripts/run_predict.py --config task_A_final/configs/final_16.yaml --input task_A_final/val_input_16.pkl --out task_A_final/runs/selector_full_val/pred_16.pkl --debug-gap-csv task_A_final/runs/selector_full_val/selector_decisions_16.csv
python task_A_final/scripts/smoke_check.py --input-8 task_A_final/val_input_8.pkl --input-16 task_A_final/val_input_16.pkl --pred-8 task_A_final/runs/selector_full_val/pred_8.pkl --pred-16 task_A_final/runs/selector_full_val/pred_16.pkl
```

### 5. Analyze final predictions

```powershell
python task_A_final/scripts/run_unified_analysis.py --input-8 task_A_final/val_input_8.pkl --input-16 task_A_final/val_input_16.pkl --pred-8 task_A_final/runs/selector_full_val/pred_8.pkl --pred-16 task_A_final/runs/selector_full_val/pred_16.pkl --gt task_A_final/val_gt.pkl --map map --run-name selector_full_val --out-dir task_A_final/runs/selector_full_val/analysis_unified
python task_A_final/scripts/run_longitudinal_analysis.py --versions baseline1=task_A_recovery/pred_linear_val_8.pkl,task_A_recovery/pred_linear_val_16.pkl baseline23e5=task_A_recovery/pred_hmm_val_8_b23_e5_gapaware.pkl,task_A_recovery/pred_hmm_val_16_b23_e5_gapaware.pkl b28_compat=task_A_final/runs/b28_compat_full/pred_8.pkl,task_A_final/runs/b28_compat_full/pred_16.pkl final=task_A_final/runs/selector_full_val/pred_8.pkl,task_A_final/runs/selector_full_val/pred_16.pkl --input-8 task_A_final/val_input_8.pkl --input-16 task_A_final/val_input_16.pkl --gt task_A_final/val_gt.pkl --map map --out-dir task_A_final/runs/longitudinal_report
```

Expected final validation metrics are approximately:

- 1/8: MAE `78.10m`, RMSE `105.38m`, topology violation `1.69%`, shape mean `30.84m`.
- 1/16: MAE `137.88m`, RMSE `189.63m`, topology violation `3.93%`, shape mean `44.67m`.

### 6. Run golden regression

```powershell
python task_A_final/scripts/run_golden_regression.py
```

### 7. Generate Task A classroom submission

```powershell
python task_A_final/scripts/run_predict.py --config task_A_final/configs/final_8.yaml --input task_A_final/test_input_8.pkl --out task_A_final/submissions/final/pred_test_8.pkl
python task_A_final/scripts/run_predict.py --config task_A_final/configs/final_16.yaml --input task_A_final/test_input_16.pkl --out task_A_final/submissions/final/pred_test_16.pkl
python task_A_final/scripts/smoke_check.py --input-8 task_A_final/test_input_8.pkl --input-16 task_A_final/test_input_16.pkl --pred-8 task_A_final/submissions/final/pred_test_8.pkl --pred-16 task_A_final/submissions/final/pred_test_16.pkl
python task_A_final/scripts/make_submission.py --pred-8 task_A_final/submissions/final/pred_test_8.pkl --pred-16 task_A_final/submissions/final/pred_test_16.pkl --out-dir task_A_final/submissions/final_bundle
```

## Task B Reproduction

Run from the repository root.

```powershell
python task_B_tte/baseline_tte.py train --hgb-max-iter 450 --model-out task_B_tte/model_baseline_hgb.pkl --metrics-out task_B_tte/metrics_baseline_hgb.json --val-pred-out task_B_tte/pred_val_baseline_hgb.pkl
python task_B_tte/phase4_residual_ensemble.py train --model-out task_B_tte/model_phase4_residual_ensemble.pkl --metrics-out task_B_tte/metrics_phase4_residual_ensemble.json --val-pred-out task_B_tte/pred_val_phase4_residual_ensemble.pkl
python task_B_tte/analyze_tte.py --pred task_B_tte/pred_val_phase4_residual_ensemble.pkl --gt task_B_tte/val_gt.pkl --input task_B_tte/val_input.pkl --output-dir task_B_tte/analysis_outputs_phase4_residual_ensemble --reference-metrics task_B_tte/analysis_outputs_baseline_hgb/global_metrics.json --milestone-name phase4_residual_ensemble_v1
```

Expected Phase4 validation metrics are approximately `MAE=16.3747s`, `RMSE=25.4080s`, `MAPE=1.4102%`.

Generate the classroom submission:

```powershell
python task_B_tte/scripts/run_onsite.py --input task_B_tte/test_input.pkl
```

Manual equivalent:

```powershell
python task_B_tte/phase4_residual_ensemble.py predict --model-in task_B_tte/model_phase4_residual_ensemble.pkl --input task_B_tte/test_input.pkl --output task_B_tte/submissions/onsite_primary/pred_test.pkl
python task_B_tte/scripts/smoke_check.py --input task_B_tte/test_input.pkl --pred task_B_tte/submissions/onsite_primary/pred_test.pkl
python task_B_tte/scripts/make_submission.py --pred task_B_tte/submissions/onsite_primary/pred_test.pkl --out-dir task_B_tte/submissions/onsite_primary_bundle
```

## Report Artifacts

The main report is `project_report.md`. Stable report figures are stored in `docs/report_figures/`; key verification notes are in `docs/report_audit.md`.

Markdown image links can be checked with:

```powershell
$text = Get-Content -Raw project_report.md
[regex]::Matches($text, '!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)') | ForEach-Object {
  $p = if ($_.Groups[1].Value) { $_.Groups[1].Value } else { $_.Groups[2].Value }
  if ($p -and -not ($p -match '^(https?://|#)') -and -not (Test-Path -LiteralPath $p)) { $p }
}
```

## Release Branches

- `release/full-artifacts`: complete archival branch with map, core data, report-referenced intermediate runs, analysis outputs, and final artifacts.
- `main`: reproducible release branch with code, docs, report, map, core data, and golden fixtures; generated runs/caches/submissions are removed.
- `release/code-lite`: lightweight code-reading branch with no data, no map, no generated outputs, and no model artifacts.
