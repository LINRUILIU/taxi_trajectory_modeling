# Task B Onsite Checklist

1. Confirm the classroom file `test_input.pkl` is readable.
2. Default primary path: run the `phase4_residual_ensemble` model.
3. Default fallback path: prepare a second bundle with `baseline_hgb`.
4. Run `scripts/smoke_check.py` on every generated `pred_test.pkl`.
5. Ensure smoke check covers record count, `traj_id` order, duplicate IDs, and positive finite `travel_time`.
6. Run `scripts/make_submission.py` to prepare the primary bundle.
7. Keep the fallback bundle ready before final upload.
8. Verify the chosen bundle contains `pred_test.pkl` and `submission_manifest.md`.

## Rollback Triggers

- primary prediction command fails
- smoke check reports count mismatch, `traj_id` mismatch, duplicate IDs, or invalid `travel_time`
- model artifact path is missing or unreadable
- a map-enabled baseline model is selected onsite but the required OSM/cache path is unavailable
