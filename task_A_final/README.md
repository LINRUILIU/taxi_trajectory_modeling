# Task A Final Pipeline

`task_A_final` is the reproducible submission pipeline for Task A. Historical exploration stays in `task_A_recovery`; all final-facing commands should run from here.

## Layout

- `configs/`: strategy configs and batch manifests
- `src/taska/`: reusable prediction and analysis logic
- `scripts/`: CLI entry points
- `caches/`: graph and overlay caches
- `runs/`: experiment outputs, logs, metadata, config snapshots
- `submissions/`: final submission files
- `golden/`: small-sample regression assets
- `docs/`: notes, logs, onsite checklist

## Phase 0: PCHIP-only closed loop

```bash
python scripts/run_predict.py --config configs/exp_pchip_only_8.yaml --input val_input_8.pkl --out runs/pchip_only_val/pred_8.pkl
python scripts/run_predict.py --config configs/exp_pchip_only_16.yaml --input val_input_16.pkl --out runs/pchip_only_val/pred_16.pkl
python scripts/smoke_check.py --input-8 val_input_8.pkl --input-16 val_input_16.pkl --pred-8 runs/pchip_only_val/pred_8.pkl --pred-16 runs/pchip_only_val/pred_16.pkl
python scripts/make_submission.py --pred-8 runs/pchip_only_val/pred_8.pkl --pred-16 runs/pchip_only_val/pred_16.pkl --out-dir submissions/pchip_only_val
```

## Validation analysis

```bash
python scripts/run_analyze.py --input-8 val_input_8.pkl --input-16 val_input_16.pkl --pred-8 runs/pchip_only_val/pred_8.pkl --pred-16 runs/pchip_only_val/pred_16.pkl --gt val_gt.pkl --out-dir runs/pchip_only_val/analysis
python scripts/run_static_eda.py --input-8 val_input_8.pkl --input-16 val_input_16.pkl --out-dir runs/static_eda
```

## B28 wrapper baseline

```bash
python scripts/run_predict.py --config configs/exp_b28_compat_8.yaml --input val_input_8.pkl --out runs/b28_compat_val/pred_8.pkl
python scripts/run_predict.py --config configs/exp_b28_compat_16.yaml --input val_input_16.pkl --out runs/b28_compat_val/pred_16.pkl
python scripts/run_analyze.py --input-8 val_input_8.pkl --input-16 val_input_16.pkl --pred-8 runs/b28_compat_val/pred_8.pkl --pred-16 runs/b28_compat_val/pred_16.pkl --gt val_gt.pkl --out-dir runs/b28_compat_val/analysis
```

## Onsite flow

```bash
python scripts/run_predict.py --config configs/final_8.yaml --input test_input_8.pkl --out submissions/final/pred_test_8.pkl
python scripts/run_predict.py --config configs/final_16.yaml --input test_input_16.pkl --out submissions/final/pred_test_16.pkl
python scripts/smoke_check.py --input-8 test_input_8.pkl --input-16 test_input_16.pkl --pred-8 submissions/final/pred_test_8.pkl --pred-16 submissions/final/pred_test_16.pkl
python scripts/make_submission.py --pred-8 submissions/final/pred_test_8.pkl --pred-16 submissions/final/pred_test_16.pkl --out-dir submissions/final_bundle
```

## Golden regression

```bash
python scripts/run_golden_regression.py
```

## Troubleshooting

- If `b28_compat` is slow on first run, let it build `caches/map_graph_cache.pkl`.
- `smoke_check.py` does not require GT or OSM; use it for onsite validation.
- Every prediction run stores `metadata.json` and a config snapshot next to the output file.
